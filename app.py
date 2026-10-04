
import streamlit as st
import pandas as pd
import sqlite3
from groq import Groq

st.set_page_config(page_title="AI SQL Analytics Agent", layout="wide")

@st.cache_resource
def load_resources():
    conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
    client = Groq()
    return conn, client

conn, client = load_resources()

schema_description = """
Tables:

customers (customer_id, name, city, signup_date, segment)
products (product_id, name, category, price)
orders (order_id, customer_id, order_date, status)
order_items (order_item_id, order_id, product_id, quantity, unit_price)
returns (return_id, order_item_id, return_date, reason)
reviews (review_id, product_id, customer_id, rating, review_date)

Relationships:
- orders.customer_id -> customers.customer_id
- order_items.order_id -> orders.order_id
- order_items.product_id -> products.product_id
- returns.order_item_id -> order_items.order_item_id
- reviews.product_id -> products.product_id
- reviews.customer_id -> customers.customer_id

IMPORTANT DEFINITION: "revenue" and "total revenue" mean NET revenue -
i.e. revenue from order_items EXCLUDING any order_item that appears in the
returns table (a returned item generated no final revenue). Always exclude
returned order_items (via a LEFT JOIN ... IS NULL, or NOT IN a subquery
against returns.order_item_id) when computing any revenue figure, unless
the question explicitly asks for gross/pre-return revenue.
"""

dataframe_schema = """
customers_df columns: customer_id, name, city, signup_date, segment
products_df columns: product_id, name, category, price
orders_df columns: order_id, customer_id, order_date, status
order_items_df columns: order_item_id, order_id, product_id, quantity, unit_price
returns_df columns: return_id, order_item_id, return_date, reason
reviews_df columns: review_id, product_id, customer_id, rating, review_date

IMPORTANT: the returns_df table records THAT an order_item was returned
(one row per return event). It does NOT have a return_quantity column or
any partial-quantity tracking. To count returns, count rows or use
order_item_id membership - do not assume a quantity field exists on
returns_df.

IMPORTANT DEFINITION: "revenue" and "total revenue" mean NET revenue -
exclude any order_item whose order_item_id appears in returns_df, unless
the question explicitly asks for gross/pre-return revenue.
"""

def generate_sql(question):
    prompt = f"""You are a SQL analyst. Given this database schema:

{schema_description}

Write a single SQLite query that answers this question: "{question}"

Return ONLY the SQL query, no explanation, no markdown formatting, no backticks.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

def self_check(question, sql, result, error):
    review_prompt = f"""You wrote this SQL to answer the question: "{question}"

SQL:
{sql}

{"It failed with this error: " + error if error else "It ran and returned this result: " + result.to_string()}

Review your own query critically. Check specifically for:
- Aggregating at the wrong level of granularity
- Missing GROUP BY causing duplicate counting after a JOIN
- Using INNER JOIN when LEFT JOIN was needed
- Whether revenue figures correctly exclude returned order_items
- Whether the result plausibly answers the question asked

Respond with ONLY one of:
"OK" if correct
"RETRY: <specific reason>" if something is wrong
"""
    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": review_prompt}]
    ).choices[0].message.content.strip()

def fix_sql(question, old_sql, reason):
    fix_prompt = f"""Your previous SQL for the question "{question}" had this issue: {reason}

Previous SQL:
{old_sql}

Write a corrected SQLite query. Return ONLY the SQL, no explanation, no markdown.
"""
    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": fix_prompt}]
    ).choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

def dual_verify(question, conn):
    pandas_prompt = f"""Given these pandas DataFrames, already loaded, with EXACTLY these columns:

{dataframe_schema}

Write pandas code (not SQL) that answers this question: "{question}"

Only use columns listed above. Do not assume a column exists if it is not
listed. The code must end by assigning the final answer to a variable
called pandas_answer.
Return ONLY the Python code, no explanation, no markdown, no backticks.
"""
    pandas_code = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": pandas_prompt}]
    ).choices[0].message.content.strip().replace("```python", "").replace("```", "").strip()

    customers_df = pd.read_sql_query("SELECT * FROM customers", conn)
    products_df = pd.read_sql_query("SELECT * FROM products", conn)
    orders_df = pd.read_sql_query("SELECT * FROM orders", conn)
    order_items_df = pd.read_sql_query("SELECT * FROM order_items", conn)
    returns_df = pd.read_sql_query("SELECT * FROM returns", conn)
    reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)

    local_scope = {
        "customers_df": customers_df, "products_df": products_df,
        "orders_df": orders_df, "order_items_df": order_items_df,
        "returns_df": returns_df, "reviews_df": reviews_df, "pd": pd
    }
    try:
        exec(pandas_code, {}, local_scope)
        return local_scope.get("pandas_answer"), pandas_code, None
    except Exception as e:
        return None, pandas_code, str(e)

def answers_match(sql_result, pandas_answer):
    """
    Returns True (matches), False (mismatch), or None (could not compare).
    Handles: single-value results, DataFrame results, and dict results
    (a common shape the pandas path produces for multi-field answers).
    """
    try:
        if isinstance(pandas_answer, dict):
            if sql_result.shape[0] != 1:
                return None
            row = sql_result.iloc[0]
            numeric_keys = [k for k, v in pandas_answer.items() if isinstance(v, (int, float))]
            if not numeric_keys:
                return None
            mismatches = []
            for k in numeric_keys:
                matches_a_column = any(
                    k.lower() in str(col).lower() or str(col).lower() in k.lower()
                    for col in sql_result.columns
                )
                if not matches_a_column:
                    continue
                best_col = min(
                    sql_result.columns,
                    key=lambda c: abs(len(str(c)) - len(k))
                )
                try:
                    sql_val = float(row[best_col])
                    pd_val = float(pandas_answer[k])
                    if abs(sql_val - pd_val) > max(0.01, abs(sql_val) * 0.001):
                        mismatches.append((k, sql_val, pd_val))
                except Exception:
                    continue
            if not mismatches:
                return None if not numeric_keys else True
            return False

        elif isinstance(pandas_answer, pd.DataFrame):
            sql_sorted = sql_result.reset_index(drop=True)
            pd_sorted = pandas_answer.reset_index(drop=True)
            return sql_sorted.equals(pd_sorted) or (
                sql_sorted.select_dtypes(include="number").round(2).equals(
                    pd_sorted.select_dtypes(include="number").round(2)
                )
            )
        else:
            sql_val = sql_result.iloc[0, 0]
            return abs(float(sql_val) - float(pandas_answer)) < 0.01
    except Exception:
        return None

st.title("AI SQL Analytics Agent")
st.caption("Ask a business question about the e-commerce database. The agent writes SQL, checks its own work, and verifies the answer independently with pandas.")

question = st.text_input("Your question")

if st.button("Ask") and question:
    with st.spinner("Generating SQL..."):
        sql = generate_sql(question)

    try:
        result = pd.read_sql_query(sql, conn)
        error = None
    except Exception as e:
        result = None
        error = str(e)

    with st.spinner("Reviewing the query..."):
        review = self_check(question, sql, result, error)

    retried = False
    if review.startswith("RETRY"):
        retried = True
        with st.spinner("Fixing the query..."):
            sql = fix_sql(question, sql, review)
        try:
            result = pd.read_sql_query(sql, conn)
            error = None
        except Exception as e:
            result = None
            error = str(e)

    st.markdown("### SQL used")
    if retried:
        st.warning(f"Self-correction triggered: {review}")
    st.code(sql, language="sql")

    if error:
        st.error(f"Query failed: {error}")
    else:
        st.markdown("### Answer")
        st.dataframe(result)

        with st.spinner("Verifying independently with pandas..."):
            pandas_answer, pandas_code, pandas_error = dual_verify(question, conn)

        with st.expander("Independent verification (pandas)", expanded=True):
            st.code(pandas_code, language="python")
            if pandas_error:
                st.warning(f"Pandas verification could not run: {pandas_error}")
            else:
                match = answers_match(result, pandas_answer)
                if match is True:
                    st.success("Verified: independent pandas computation matches the SQL result.")
                elif match is False:
                    st.error("Mismatch: the independent pandas computation does not match the SQL result. Treat this answer with caution.")
                else:
                    st.info("Could not automatically compare the two results - review both manually.")
                st.write("Pandas result:")
                st.write(pandas_answer)

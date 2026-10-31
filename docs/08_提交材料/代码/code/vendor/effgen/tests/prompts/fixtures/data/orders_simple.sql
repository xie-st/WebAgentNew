-- Simple orders + customers schema used for fixture tests

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        VARCHAR(120) NOT NULL,
    country     VARCHAR(60)  NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date  DATE    NOT NULL,
    total_amt   NUMERIC(12,2) NOT NULL
);

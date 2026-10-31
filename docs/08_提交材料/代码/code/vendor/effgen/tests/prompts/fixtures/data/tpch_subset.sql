-- TPC-H subset schema (public benchmark, freely available)
-- Subset: orders + lineitem + customer tables

CREATE TABLE customer (
    c_custkey    INTEGER       PRIMARY KEY,
    c_name       VARCHAR(25)   NOT NULL,
    c_address    VARCHAR(40)   NOT NULL,
    c_nationkey  INTEGER       NOT NULL,
    c_phone      CHAR(15)      NOT NULL,
    c_acctbal    NUMERIC(15,2) NOT NULL,
    c_mktsegment CHAR(10)      NOT NULL,
    c_comment    VARCHAR(117)  NOT NULL
);

CREATE TABLE orders (
    o_orderkey      INTEGER       PRIMARY KEY,
    o_custkey       INTEGER       NOT NULL REFERENCES customer(c_custkey),
    o_orderstatus   CHAR(1)       NOT NULL,
    o_totalprice    NUMERIC(15,2) NOT NULL,
    o_orderdate     DATE          NOT NULL,
    o_orderpriority CHAR(15)      NOT NULL,
    o_clerk         CHAR(15)      NOT NULL,
    o_shippriority  INTEGER       NOT NULL,
    o_comment       VARCHAR(79)   NOT NULL
);

CREATE TABLE lineitem (
    l_orderkey      INTEGER       NOT NULL REFERENCES orders(o_orderkey),
    l_partkey       INTEGER       NOT NULL,
    l_suppkey       INTEGER       NOT NULL,
    l_linenumber    INTEGER       NOT NULL,
    l_quantity      NUMERIC(15,2) NOT NULL,
    l_extendedprice NUMERIC(15,2) NOT NULL,
    l_discount      NUMERIC(15,2) NOT NULL,
    l_tax           NUMERIC(15,2) NOT NULL,
    l_returnflag    CHAR(1)       NOT NULL,
    l_linestatus    CHAR(1)       NOT NULL,
    l_shipdate      DATE          NOT NULL,
    l_commitdate    DATE          NOT NULL,
    l_receiptdate   DATE          NOT NULL,
    l_shipinstruct  CHAR(25)      NOT NULL,
    l_shipmode      CHAR(10)      NOT NULL,
    l_comment       VARCHAR(44)   NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);

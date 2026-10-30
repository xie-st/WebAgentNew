"""Property and regression tests for workflow YAML loading.

``WorkflowDAG.from_yaml`` reads a file a user wrote by hand, so every shape a
hand-written file can take reaches it: an empty file, a bare list, a ``nodes:``
block that is a string, a node with no ``id``, a ``depends_on`` naming a node that
does not exist, a cycle.

Asserts that:
  1. A file that is not a workflow raises ``ValueError`` naming the file and the
     offending node or edge — never ``AttributeError``/``TypeError``/``KeyError``
     from inside the parser.
  2. A file that *is* a workflow builds exactly the graph it declares: the same
     node ids, in order, and one edge per declared dependency. Nothing is dropped
     and nothing is invented.
  3. The two ways of declaring wiring (per-node ``depends_on`` and a top-level
     ``edges`` list) build the same graph.
  4. A single dependency written without list syntax (``depends_on: search``) is
     that one dependency, not one per character.
"""

from __future__ import annotations

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.core.workflow import WorkflowDAG

pytestmark = pytest.mark.fuzz


def _write(tmp_path, document) -> str:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return str(path)


# ---------------------------------------------------------------------------
# Refusals name the file and the position
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (None, "empty"),
        ([], "expected a workflow mapping"),
        ([{"id": "a"}], "expected a workflow mapping"),
        ("just text", "expected a workflow mapping"),
        (7, "expected a workflow mapping"),
        ({"workflow": None}, "'workflow' must be a mapping"),
        ({"workflow": "text"}, "'workflow' must be a mapping"),
        ({"workflow": []}, "'workflow' must be a mapping"),
        ({"name": ["a"]}, "'name' must be text"),
        ({"nodes": "abc"}, "'nodes' must be a list"),
        ({"nodes": 5}, "'nodes' must be a list"),
        ({"nodes": {"a": 1}}, "'nodes' must be a list"),
        ({"nodes": [None]}, "node 1 must be a mapping"),
        ({"nodes": ["text"]}, "node 1 must be a mapping"),
        ({"nodes": [5]}, "node 1 must be a mapping"),
        ({"nodes": [{}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"agent": "x"}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"id": None}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"id": ""}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"id": "  "}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"id": []}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"id": {"a": 1}}]}, "node 1 needs a non-empty 'id'"),
        ({"nodes": [{"id": "a"}, {"id": "a"}]}, "Duplicate node id"),
        ({"nodes": [{"id": "a", "tools": {"a": 1}}]}, "tools must be a list of names"),
        (
            {"nodes": [{"id": "a", "depends_on": {"x": 1}}]},
            "depends_on must be a list of names",
        ),
        (
            {"nodes": [{"id": "a", "depends_on": ["nope"]}]},
            "not in DAG",
        ),
        ({"nodes": [{"id": "a"}], "edges": "a,b"}, "'edges' must be a list"),
        ({"nodes": [{"id": "a"}], "edges": 3}, "'edges' must be a list"),
        ({"nodes": [{"id": "a"}, {"id": "b"}], "edges": [["a"]]}, "edge 1"),
        (
            {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a"}]},
            "edge 1",
        ),
        ({"nodes": [{"id": "a"}], "edges": [["a", "a"]]}, "edge 1"),
        (
            {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [["a", "b"], ["b", "a"]]},
            "edge 2",
        ),
    ],
)
def test_a_file_that_is_not_a_workflow_is_refused_by_name(
    tmp_path, document, expected
) -> None:
    path = _write(tmp_path, document)
    with pytest.raises(ValueError) as excinfo:
        WorkflowDAG.from_yaml(path)
    message = str(excinfo.value)
    assert expected in message, message
    assert path in message, message


@settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    st.recursive(
        st.none()
        | st.booleans()
        | st.integers(min_value=-100, max_value=100)
        | st.text(max_size=10),
        lambda children: st.lists(children, max_size=3)
        | st.dictionaries(
            st.sampled_from(["id", "nodes", "edges", "workflow", "depends_on",
                             "tools", "name", "output_key", "input_keys"]),
            children,
            max_size=4,
        ),
        max_leaves=10,
    )
)
def test_arbitrary_yaml_never_escapes_as_a_raw_crash(tmp_path, document) -> None:
    """Any document either builds a DAG or is refused with a ``ValueError``."""
    path = _write(tmp_path, document)
    try:
        dag = WorkflowDAG.from_yaml(path)
    except ValueError:
        return
    # A DAG that loaded must be usable, not merely constructed.
    assert isinstance(dag.topological_order(), list)
    assert isinstance(dag.to_dict(), dict)


# ---------------------------------------------------------------------------
# What the file declares is what the graph holds
# ---------------------------------------------------------------------------

_IDS = st.lists(
    st.text(alphabet="abcdefghij", min_size=1, max_size=4),
    min_size=1,
    max_size=6,
    unique=True,
)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(_IDS)
def test_a_chain_loads_as_exactly_the_chain_it_declares(tmp_path, ids) -> None:
    """Every declared node survives, in order, with one edge per dependency."""
    nodes = [{"id": ids[0]}] + [
        {"id": nid, "depends_on": [ids[i]]} for i, nid in enumerate(ids[1:])
    ]
    path = _write(tmp_path, {"workflow": {"name": "chain", "nodes": nodes}})
    dag = WorkflowDAG.from_yaml(path)

    assert [n.id for n in dag.nodes] == ids
    assert len(dag.edges) == len(ids) - 1
    assert dag.topological_order() == ids
    assert dag.entry_nodes() == [ids[0]]


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(_IDS)
def test_depends_on_and_an_edges_block_build_the_same_graph(tmp_path, ids) -> None:
    left_dir = tmp_path / "depends"
    right_dir = tmp_path / "edges"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    via_depends = _write(
        left_dir,
        {
            "nodes": [{"id": ids[0]}]
            + [{"id": n, "depends_on": [ids[i]]} for i, n in enumerate(ids[1:])]
        },
    )
    via_edges = _write(
        right_dir,
        {
            "nodes": [{"id": n} for n in ids],
            "edges": [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)],
        },
    )
    left = WorkflowDAG.from_yaml(via_depends)
    right = WorkflowDAG.from_yaml(via_edges)

    assert [n.id for n in left.nodes] == [n.id for n in right.nodes]
    assert {(e.source, e.target) for e in left.edges} == {
        (e.source, e.target) for e in right.edges
    }


def test_a_single_dependency_without_list_syntax_is_one_dependency(tmp_path) -> None:
    """``depends_on: search`` is one edge, not one per character."""
    path = _write(
        tmp_path,
        {"nodes": [{"id": "search"}, {"id": "summarize", "depends_on": "search"}]},
    )
    dag = WorkflowDAG.from_yaml(path)
    assert [(e.source, e.target) for e in dag.edges] == [("search", "summarize")]


def test_a_single_tool_without_list_syntax_is_one_tool(tmp_path) -> None:
    path = _write(tmp_path, {"nodes": [{"id": "a", "tools": "web_search"}]})
    dag = WorkflowDAG.from_yaml(path)
    assert dag.get_node("a").tools == ["web_search"]


def test_a_numeric_node_id_addresses_the_same_node(tmp_path) -> None:
    """YAML reads ``id: 1`` as an int; the id and every reference to it agree."""
    path = _write(
        tmp_path, {"nodes": [{"id": 1}, {"id": 2, "depends_on": [1]}]}
    )
    dag = WorkflowDAG.from_yaml(path)
    assert [n.id for n in dag.nodes] == ["1", "2"]
    assert [(e.source, e.target) for e in dag.edges] == [("1", "2")]


def test_output_key_defaults_to_the_node_id(tmp_path) -> None:
    path = _write(tmp_path, {"nodes": [{"id": "a"}, {"id": "b", "output_key": "z"}]})
    dag = WorkflowDAG.from_yaml(path)
    assert dag.get_node("a").output_key == "a"
    assert dag.get_node("b").output_key == "z"


def test_an_empty_nodes_block_is_an_empty_workflow(tmp_path) -> None:
    """A declared-but-empty ``nodes:`` is a workflow with no nodes, not a crash."""
    for document in ({"nodes": []}, {"nodes": None}, {"workflow": {"name": "x"}}):
        dag = WorkflowDAG.from_yaml(_write(tmp_path, document))
        assert dag.nodes == []
        assert dag.topological_order() == []


def test_unparseable_yaml_still_raises_a_typed_parse_error(tmp_path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text("nodes: [unclosed\n  bad: : :", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        WorkflowDAG.from_yaml(str(path))

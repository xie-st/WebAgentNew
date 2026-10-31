"""Tests for ToolFallbackChain."""

from effgen.tools.fallback import ToolFallbackChain


class TestToolFallbackChainDefaults:
    """Test default fallback chains."""

    def test_default_chains_exist(self):
        chain = ToolFallbackChain()
        assert chain.has_fallbacks("calculator")
        assert chain.has_fallbacks("python_repl")
        assert chain.has_fallbacks("code_executor")
        assert chain.has_fallbacks("web_search")

    def test_calculator_fallbacks(self):
        chain = ToolFallbackChain()
        fallbacks = chain.get_fallbacks("calculator")
        assert "python_repl" in fallbacks
        assert "code_executor" in fallbacks

    def test_no_fallback_returns_empty(self):
        chain = ToolFallbackChain()
        assert chain.get_fallbacks("nonexistent_tool") == []

    def test_has_fallbacks_false_for_unknown(self):
        chain = ToolFallbackChain()
        assert chain.has_fallbacks("nonexistent_tool") is False


class TestToolFallbackChainCustom:
    """Test custom fallback chains."""

    def test_custom_chains_override_defaults(self):
        custom = {"calculator": ["custom_tool"]}
        chain = ToolFallbackChain(custom_chains=custom)
        assert chain.get_fallbacks("calculator") == ["custom_tool"]

    def test_custom_chains_add_new(self):
        custom = {"my_tool": ["fallback_a", "fallback_b"]}
        chain = ToolFallbackChain(custom_chains=custom)
        assert chain.get_fallbacks("my_tool") == ["fallback_a", "fallback_b"]
        # Defaults still exist
        assert chain.has_fallbacks("calculator")

    def test_register_chain(self):
        chain = ToolFallbackChain()
        chain.register_chain("new_tool", ["fb1", "fb2"])
        assert chain.get_fallbacks("new_tool") == ["fb1", "fb2"]

    def test_register_chain_replaces(self):
        chain = ToolFallbackChain()
        chain.register_chain("calculator", ["only_this"])
        assert chain.get_fallbacks("calculator") == ["only_this"]


class TestToolFallbackChainImmutability:
    """Test that returned lists are copies."""

    def test_get_fallbacks_returns_copy(self):
        chain = ToolFallbackChain()
        fallbacks = chain.get_fallbacks("calculator")
        fallbacks.append("hacked")
        assert "hacked" not in chain.get_fallbacks("calculator")

    def test_register_chain_copies_input(self):
        chain = ToolFallbackChain()
        original = ["a", "b"]
        chain.register_chain("t", original)
        original.append("c")
        assert chain.get_fallbacks("t") == ["a", "b"]

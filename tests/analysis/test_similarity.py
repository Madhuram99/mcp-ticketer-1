"""Tests for ticket similarity detection and hybrid pipeline."""

from datetime import datetime

import numpy as np
import pytest

from mcp_ticketer.analysis.similarity import (
    BM25_AVAILABLE,
    HYBRID_AVAILABLE,
    SEMANTIC_AVAILABLE,
    HybridSimilarityPipeline,
    TicketSimilarityAnalyzer,
    _normalize_matrix,
    _tokenize,
)
from mcp_ticketer.core.models import Priority, Task, TicketState


@pytest.fixture
def sample_tickets():
    """Create sample tickets for testing."""
    return [
        Task(
            id="TICKET-1",
            title="Fix login authentication bug",
            description="Users cannot log in with SSO credentials",
            priority=Priority.HIGH,
            state=TicketState.OPEN,
            tags=["bug", "authentication"],
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 15),
        ),
        Task(
            id="TICKET-2",
            title="Fix authentication login issue",
            description="SSO login is not working for users",
            priority=Priority.HIGH,
            state=TicketState.OPEN,
            tags=["bug", "authentication", "sso"],
            created_at=datetime(2024, 1, 2),
            updated_at=datetime(2024, 1, 16),
        ),
        Task(
            id="TICKET-3",
            title="Add user profile page",
            description="Create a new profile page for user settings",
            priority=Priority.MEDIUM,
            state=TicketState.OPEN,
            tags=["feature", "ui"],
            created_at=datetime(2024, 1, 3),
            updated_at=datetime(2024, 1, 17),
        ),
        Task(
            id="TICKET-4",
            title="Implement user settings interface",
            description="Build interface for users to manage their settings",
            priority=Priority.MEDIUM,
            state=TicketState.OPEN,
            tags=["feature", "ui", "settings"],
            created_at=datetime(2024, 1, 4),
            updated_at=datetime(2024, 1, 18),
        ),
        Task(
            id="TICKET-5",
            title="Update documentation",
            description="Update API documentation for new endpoints",
            priority=Priority.LOW,
            state=TicketState.OPEN,
            tags=["documentation"],
            created_at=datetime(2024, 1, 5),
            updated_at=datetime(2024, 1, 19),
        ),
    ]


class TestTicketSimilarityAnalyzer:
    """Test cases for TicketSimilarityAnalyzer."""

    def test_initialization(self) -> None:
        """Test analyzer initialization with default parameters."""
        analyzer = TicketSimilarityAnalyzer()
        assert analyzer.threshold == 0.75
        assert analyzer.title_weight == 0.7
        assert analyzer.description_weight == 0.3

    def test_custom_initialization(self) -> None:
        """Test analyzer initialization with custom parameters."""
        analyzer = TicketSimilarityAnalyzer(
            threshold=0.8,
            title_weight=0.6,
            description_weight=0.4,
        )
        assert analyzer.threshold == 0.8
        assert analyzer.title_weight == 0.6
        assert analyzer.description_weight == 0.4

    def test_invalid_weights_rejected(self) -> None:
        """Test that invalid weight combinations raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            TicketSimilarityAnalyzer(keyword_weight=0.8, semantic_weight=0.8)
        with pytest.raises(ValueError, match="non-negative"):
            TicketSimilarityAnalyzer(keyword_weight=-0.3, semantic_weight=1.3)

    def test_pipeline_auto_mode(self) -> None:
        """Test that auto mode selects hybrid when BM25 available."""
        analyzer = TicketSimilarityAnalyzer(pipeline="auto")
        if HYBRID_AVAILABLE:
            assert analyzer.pipeline == "hybrid"
        else:
            assert analyzer.pipeline == "classic"

    def test_pipeline_classic_mode(self) -> None:
        """Test explicit classic pipeline mode."""
        analyzer = TicketSimilarityAnalyzer(pipeline="classic")
        assert analyzer.pipeline == "classic"
        assert analyzer._hybrid_pipeline is None

    def test_pipeline_hybrid_mode(self) -> None:
        """Test explicit hybrid pipeline mode."""
        analyzer = TicketSimilarityAnalyzer(pipeline="hybrid")
        assert analyzer.pipeline == "hybrid"
        if HYBRID_AVAILABLE:
            assert analyzer._hybrid_pipeline is not None

    def test_pipeline_weights(self) -> None:
        """Test custom pipeline weights."""
        analyzer = TicketSimilarityAnalyzer(
            pipeline="hybrid",
            keyword_weight=0.5,
            semantic_weight=0.5,
        )
        assert analyzer.keyword_weight == 0.5
        assert analyzer.semantic_weight == 0.5

    def test_pipeline_info_property(self) -> None:
        """Test pipeline_info returns correct metadata."""
        analyzer = TicketSimilarityAnalyzer(pipeline="classic")
        info = analyzer.pipeline_info
        assert info["pipeline"] == "classic"
        assert isinstance(info["bm25_available"], bool)
        assert isinstance(info["semantic_available"], bool)
        assert isinstance(info["hybrid_available"], bool)

    def test_pipeline_info_hybrid(self) -> None:
        """Test pipeline_info for hybrid mode includes stage details."""
        analyzer = TicketSimilarityAnalyzer(
            pipeline="hybrid",
            keyword_weight=0.4,
            semantic_weight=0.6,
        )
        info = analyzer.pipeline_info
        assert info["pipeline"] == "hybrid"
        if analyzer._hybrid_pipeline:
            assert "active_stages" in info
            assert info["keyword_weight"] == 0.4
            assert info["semantic_weight"] == 0.6

    def test_find_similar_tickets_result_structure(self, sample_tickets) -> None:
        """Test that results have correct structure and respect threshold."""
        analyzer = TicketSimilarityAnalyzer(threshold=0.2, pipeline="classic")
        results = analyzer.find_similar_tickets(sample_tickets)

        # Every result must have valid fields and score above threshold
        for result in results:
            assert result.ticket1_id.startswith("TICKET-")
            assert result.ticket2_id.startswith("TICKET-")
            assert result.similarity_score >= 0.2
            assert result.suggested_action in ("merge", "link", "ignore")
            assert 0.0 <= result.confidence <= 1.0

    def test_find_similar_tickets_target(self, sample_tickets) -> None:
        """Test finding tickets similar to a specific target."""
        analyzer = TicketSimilarityAnalyzer(threshold=0.3, pipeline="classic")
        target = sample_tickets[0]  # TICKET-1
        results = analyzer.find_similar_tickets(sample_tickets, target)

        # All results must reference the target ticket
        for result in results:
            assert result.ticket1_id == target.id

    def test_identical_tickets_detected(self) -> None:
        """Test that identical tickets are always detected as similar."""
        tickets = [
            Task(
                id="TICKET-1",
                title="Fix authentication bug",
                description="Auth bug details",
                priority=Priority.HIGH,
                state=TicketState.OPEN,
            ),
            Task(
                id="TICKET-2",
                title="Fix authentication bug",
                description="Auth bug details",
                priority=Priority.LOW,
                state=TicketState.OPEN,
            ),
        ]
        analyzer = TicketSimilarityAnalyzer(threshold=0.5, pipeline="classic")
        results = analyzer.find_similar_tickets(tickets)

        assert len(results) == 1
        assert results[0].similarity_score > 0.8
        assert results[0].ticket1_id == "TICKET-1"
        assert results[0].ticket2_id == "TICKET-2"

    def test_suggested_actions(self, sample_tickets) -> None:
        """Test that suggested actions are appropriate for similarity scores."""
        analyzer = TicketSimilarityAnalyzer(threshold=0.4, pipeline="classic")
        results = analyzer.find_similar_tickets(sample_tickets)

        for result in results:
            if result.similarity_score > 0.9:
                assert result.suggested_action == "merge"
            elif result.similarity_score > 0.75:
                assert result.suggested_action == "link"
            else:
                assert result.suggested_action == "ignore"

    def test_similarity_reasons_populated(self, sample_tickets) -> None:
        """Test that similarity reasons contain meaningful entries."""
        analyzer = TicketSimilarityAnalyzer(threshold=0.3, pipeline="classic")
        results = analyzer.find_similar_tickets(sample_tickets)

        for result in results:
            assert isinstance(result.similarity_reasons, list)
            # All results between OPEN tickets should note same_state
            assert "same_state" in result.similarity_reasons

    def test_empty_tickets_list(self) -> None:
        """Test handling of empty tickets list."""
        analyzer = TicketSimilarityAnalyzer()
        results = analyzer.find_similar_tickets([])
        assert results == []

    def test_single_ticket(self, sample_tickets) -> None:
        """Test handling of single ticket."""
        analyzer = TicketSimilarityAnalyzer()
        results = analyzer.find_similar_tickets([sample_tickets[0]])
        assert results == []

    def test_limit_parameter(self, sample_tickets) -> None:
        """Test that limit parameter is respected."""
        analyzer = TicketSimilarityAnalyzer(threshold=0.3, pipeline="classic")
        results = analyzer.find_similar_tickets(sample_tickets, limit=2)
        assert len(results) <= 2

    def test_tickets_with_no_description(self) -> None:
        """Test that title-only tickets still produce similarity results."""
        tickets = [
            Task(
                id="TICKET-1",
                title="Fix bug in login",
                description=None,
                priority=Priority.HIGH,
                state=TicketState.OPEN,
            ),
            Task(
                id="TICKET-2",
                title="Fix bug in login",
                description=None,
                priority=Priority.HIGH,
                state=TicketState.OPEN,
            ),
        ]

        analyzer = TicketSimilarityAnalyzer(threshold=0.3, pipeline="classic")
        results = analyzer.find_similar_tickets(tickets)

        # Identical titles should always be detected
        assert len(results) == 1
        assert results[0].similarity_score > 0.8

    def test_confidence_score(self, sample_tickets) -> None:
        """Test that confidence score matches similarity score."""
        analyzer = TicketSimilarityAnalyzer(threshold=0.5, pipeline="classic")
        results = analyzer.find_similar_tickets(sample_tickets)

        for result in results:
            assert result.confidence == result.similarity_score

    def test_classic_and_hybrid_both_find_identical_tickets(self) -> None:
        """Test that both pipelines find identical tickets as similar."""
        tickets = [
            Task(
                id="T-1",
                title="Fix authentication bug in SSO module",
                description="Auth bug details for the SSO login module",
                priority=Priority.HIGH,
                state=TicketState.OPEN,
            ),
            Task(
                id="T-2",
                title="Fix authentication bug in SSO module",
                description="Auth bug details for the SSO login module",
                priority=Priority.HIGH,
                state=TicketState.OPEN,
            ),
            Task(
                id="T-3",
                title="Update API documentation for REST endpoints",
                description="Comprehensive docs update for all REST API endpoints",
                priority=Priority.LOW,
                state=TicketState.OPEN,
            ),
        ]

        # Classic should find T-1 and T-2 as nearly identical
        classic = TicketSimilarityAnalyzer(threshold=0.5, pipeline="classic")
        classic_results = classic.find_similar_tickets(tickets)
        t1_t2_classic = [
            r for r in classic_results
            if {r.ticket1_id, r.ticket2_id} == {"T-1", "T-2"}
        ]
        assert len(t1_t2_classic) == 1
        assert t1_t2_classic[0].similarity_score > 0.8

        # Hybrid should also find T-1 and T-2 as similar (with 3 tickets for
        # better normalization than 2-ticket edge case)
        hybrid = TicketSimilarityAnalyzer(threshold=0.3, pipeline="hybrid")
        hybrid_results = hybrid.find_similar_tickets(tickets)
        t1_t2_hybrid = [
            r for r in hybrid_results
            if {r.ticket1_id, r.ticket2_id} == {"T-1", "T-2"}
        ]
        assert len(t1_t2_hybrid) == 1
        assert t1_t2_hybrid[0].similarity_score > 0.5


class TestHybridSimilarityPipeline:
    """Test cases for the HybridSimilarityPipeline."""

    def test_initialization_defaults(self) -> None:
        """Test pipeline initialization with default weights."""
        pipeline = HybridSimilarityPipeline()
        assert pipeline.keyword_weight == 0.3
        assert pipeline.semantic_weight == 0.7

    def test_initialization_custom_weights(self) -> None:
        """Test pipeline initialization with custom weights."""
        pipeline = HybridSimilarityPipeline(
            keyword_weight=0.5,
            semantic_weight=0.5,
        )
        assert pipeline.keyword_weight == 0.5
        assert pipeline.semantic_weight == 0.5

    def test_invalid_weights_rejected(self) -> None:
        """Test that invalid weight combinations raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            HybridSimilarityPipeline(keyword_weight=0.8, semantic_weight=0.8)
        with pytest.raises(ValueError, match="must equal 1.0"):
            HybridSimilarityPipeline(keyword_weight=0.3, semantic_weight=0.3)
        with pytest.raises(ValueError, match="non-negative"):
            HybridSimilarityPipeline(keyword_weight=-0.1, semantic_weight=1.1)

    def test_active_stages(self) -> None:
        """Test active_stages property reflects available deps."""
        pipeline = HybridSimilarityPipeline()
        stages = pipeline.active_stages

        if BM25_AVAILABLE:
            assert "bm25_keyword" in stages
        if SEMANTIC_AVAILABLE:
            assert "dense_semantic" in stages
        else:
            assert "tfidf_fallback" in stages
        assert "weighted_fusion" in stages

    def test_compute_similarity_matrix_shape(self) -> None:
        """Test that similarity matrix has correct shape."""
        pipeline = HybridSimilarityPipeline()
        texts = [
            "Fix login authentication bug",
            "Fix authentication login issue",
            "Update documentation",
        ]
        matrix = pipeline.compute_similarity_matrix(texts)
        assert matrix.shape == (3, 3)

    def test_compute_similarity_matrix_range(self) -> None:
        """Test that similarity scores are in [0, 1]."""
        pipeline = HybridSimilarityPipeline()
        texts = [
            "Fix login bug",
            "Fix login issue",
            "Update docs",
            "Add new feature",
        ]
        matrix = pipeline.compute_similarity_matrix(texts)
        assert matrix.min() >= 0.0
        assert matrix.max() <= 1.0

    def test_similar_texts_score_higher(self) -> None:
        """Test that similar texts get higher scores than dissimilar ones."""
        pipeline = HybridSimilarityPipeline()
        texts = [
            "Fix login authentication bug with SSO",
            "Fix authentication login issue with SSO",
            "Update API documentation for new endpoints",
        ]
        matrix = pipeline.compute_similarity_matrix(texts)

        # Texts 0 and 1 (both about auth) should be more similar than 0 and 2
        assert matrix[0, 1] > matrix[0, 2]

    def test_empty_texts(self) -> None:
        """Test pipeline with less than 2 texts."""
        pipeline = HybridSimilarityPipeline()
        matrix = pipeline.compute_similarity_matrix(["single text"])
        assert matrix.shape == (1, 1)

    def test_tfidf_fallback_stage(self) -> None:
        """Test that TF-IDF fallback produces valid similarity matrix."""
        pipeline = HybridSimilarityPipeline()
        texts = ["Fix login bug", "Fix login issue", "Update docs"]
        matrix = pipeline._tfidf_fallback(texts)
        assert matrix.shape == (3, 3)
        assert matrix.min() >= 0.0
        # Similar texts should score higher
        assert matrix[0, 1] > matrix[0, 2]

    @pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")
    def test_bm25_stage(self) -> None:
        """Test BM25 stage produces valid similarity matrix."""
        pipeline = HybridSimilarityPipeline()
        texts = [
            "Fix login authentication bug",
            "Fix authentication login issue",
            "Update documentation",
        ]
        matrix = pipeline._bm25_stage(texts)
        assert matrix.shape == (3, 3)
        assert matrix.min() >= 0.0
        assert matrix.max() <= 1.0
        # Auth texts should be more similar than auth vs docs
        assert matrix[0, 1] > matrix[0, 2]

    @pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")
    def test_bm25_symmetric(self) -> None:
        """Test that BM25 matrix is symmetric."""
        pipeline = HybridSimilarityPipeline()
        texts = ["Fix login bug", "Fix login issue", "Update docs"]
        matrix = pipeline._bm25_stage(texts)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_fusion_stage(self) -> None:
        """Test weighted score fusion with known inputs."""
        pipeline = HybridSimilarityPipeline(
            keyword_weight=0.5, semantic_weight=0.5
        )
        bm25 = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        semantic = np.array([[1.0, 0.6, 0.1], [0.6, 1.0, 0.2], [0.1, 0.2, 1.0]])
        fused = pipeline._fusion_stage(bm25, semantic)

        assert fused.shape == (3, 3)
        assert fused.min() >= 0.0
        assert fused.max() <= 1.0
        # Items 0,1 should be more similar than items 0,2
        assert fused[0, 1] > fused[0, 2]

    def test_fusion_renormalizes_when_bm25_zero(self) -> None:
        """Test that fusion uses full semantic weight when BM25 returns zeros."""
        pipeline = HybridSimilarityPipeline(
            keyword_weight=0.3, semantic_weight=0.7
        )
        zeros = np.zeros((3, 3))
        semantic = np.array(
            [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]]
        )
        fused = pipeline._fusion_stage(zeros, semantic)

        # With renormalization, fused should equal normalized semantic (weight=1.0)
        semantic_norm = _normalize_matrix(semantic)
        np.testing.assert_array_almost_equal(fused, semantic_norm)


class TestHelperFunctions:
    """Test cases for module-level helper functions."""

    def test_tokenize(self) -> None:
        """Test text tokenization."""
        tokens = _tokenize("Fix Login Bug")
        assert tokens == ["fix", "login", "bug"]

    def test_tokenize_empty(self) -> None:
        """Test tokenization of empty string."""
        tokens = _tokenize("")
        assert tokens == []

    def test_normalize_matrix_basic(self) -> None:
        """Test matrix normalization to [0, 1]."""
        matrix = np.array([[0.0, 5.0], [5.0, 10.0]])
        normalized = _normalize_matrix(matrix)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        # 5/10 = 0.5
        assert normalized[0, 1] == 0.5

    def test_normalize_matrix_uniform_positive(self) -> None:
        """Test normalization of uniform positive matrix returns ones."""
        matrix = np.array([[3.0, 3.0], [3.0, 3.0]])
        normalized = _normalize_matrix(matrix)
        np.testing.assert_array_equal(normalized, np.ones((2, 2)))

    def test_normalize_matrix_uniform_zero(self) -> None:
        """Test normalization of all-zero matrix returns zeros."""
        matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
        normalized = _normalize_matrix(matrix)
        np.testing.assert_array_equal(normalized, np.zeros((2, 2)))

    def test_normalize_matrix_already_normalized(self) -> None:
        """Test normalization of already-normalized matrix."""
        matrix = np.array([[0.0, 0.5], [0.5, 1.0]])
        normalized = _normalize_matrix(matrix)
        np.testing.assert_array_almost_equal(normalized, matrix)


class TestAvailabilityFlags:
    """Test that availability flags are correctly set."""

    def test_hybrid_available_requires_bm25(self) -> None:
        """Test HYBRID_AVAILABLE is True only when BM25 is available."""
        assert HYBRID_AVAILABLE == BM25_AVAILABLE

    def test_flags_are_booleans(self) -> None:
        """Test all flags are boolean values."""
        assert isinstance(BM25_AVAILABLE, bool)
        assert isinstance(SEMANTIC_AVAILABLE, bool)
        assert isinstance(HYBRID_AVAILABLE, bool)

"""Tests for ecFlow suite definition chunker."""

from coral.rag.chunkers.ecflow_chunker import EcflowChunker


class TestEcflowChunker:
    def setup_method(self):
        self.chunker = EcflowChunker()

    def test_single_task(self):
        content = "task run_model\n  edit ECF_JOB_CMD 'sbatch'\n  edit WALLTIME '02:00:00'\nendtask"
        chunks = self.chunker.chunk(content, "test.def")
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["node_type"] == "task"
        assert chunks[0]["metadata"]["name"] == "run_model"
        assert "sbatch" in chunks[0]["text"]

    def test_family_with_tasks(self):
        content = "family forecast\n  task prep_data\n  endtask\n  task run_model\n  endtask\nendfamily"
        chunks = self.chunker.chunk(content, "suite.def")
        # Should get family start, two tasks, and family end
        assert len(chunks) >= 2
        names = [c["metadata"].get("name", "") for c in chunks]
        assert "prep_data" in names
        assert "run_model" in names

    def test_empty_content_returns_empty(self):
        chunks = self.chunker.chunk("", "empty.def")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = self.chunker.chunk("   \n  \n  ", "blank.def")
        assert chunks == []

    def test_no_structure_returns_whole_file(self):
        content = "# This is a comment\nsome_variable = value\n"
        chunks = self.chunker.chunk(content, "script.ecf")
        assert len(chunks) == 1
        assert "comment" in chunks[0]["text"]

    def test_metadata_has_file_path(self):
        content = "task mytask\nendtask"
        chunks = self.chunker.chunk(content, "/path/to/suite.def")
        assert chunks[0]["metadata"]["file_path"] == "/path/to/suite.def"

    def test_metadata_has_ecflow_type(self):
        content = "task mytask\nendtask"
        chunks = self.chunker.chunk(content, "test.def")
        assert chunks[0]["metadata"]["type"] == "ecflow"

    def test_start_line_is_correct(self):
        content = "# header\n# comment\ntask my_task\n  edit VAR val\nendtask"
        chunks = self.chunker.chunk(content, "test.def")
        task_chunks = [c for c in chunks if c["metadata"].get("name") == "my_task"]
        assert len(task_chunks) == 1
        # task starts at line 3 (0-indexed line 2, 1-indexed line 3)
        assert task_chunks[0]["metadata"]["start_line"] == 3

    def test_multiple_families(self):
        content = (
            "family prep\n  task fetch_data\n  endtask\nendfamily\nfamily run\n  task execute\n  endtask\nendfamily"
        )
        chunks = self.chunker.chunk(content, "multi.def")
        names = [c["metadata"].get("name", "") for c in chunks]
        assert "fetch_data" in names
        assert "execute" in names

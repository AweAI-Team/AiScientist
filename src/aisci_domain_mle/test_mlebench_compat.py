from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aisci_domain_mle.contracts import InputSourceKind
from aisci_domain_mle.mlebench_compat import (
    LegacyCompetitionPreparer,
    LegacyMlebenchDownloader,
    find_legacy_mlebench_repo_root,
    resolve_competition_source,
    resolve_legacy_mlebench_repo_root,
)
from aisci_domain_mle.vendored_lite import vendored_lite_repo_root


class MlebenchCompatResolverTests(unittest.TestCase):
    def test_legacy_repo_root_prefers_explicit_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "custom-mle-bench"
            (repo_root / "mlebench").mkdir(parents=True)
            (repo_root / "mlebench" / "cli.py").write_text("print('placeholder')\n", encoding="utf-8")

            with mock.patch.dict("os.environ", {"AISCI_MLEBENCH_REPO": str(repo_root)}, clear=False):
                self.assertEqual(find_legacy_mlebench_repo_root(), repo_root.resolve())
                self.assertEqual(resolve_legacy_mlebench_repo_root(), repo_root.resolve())

    def test_legacy_repo_root_defaults_to_vendored_lite_registry(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(find_legacy_mlebench_repo_root(), vendored_lite_repo_root())
            self.assertEqual(resolve_legacy_mlebench_repo_root(), vendored_lite_repo_root())

    def test_resolve_legacy_repo_root_reports_missing_repo_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing_repo = tmp_path / "missing-mle-bench"

            with self.assertRaises(ValueError) as context:
                resolve_legacy_mlebench_repo_root(missing_repo)

            self.assertIn("AISCI_MLEBENCH_REPO", str(context.exception))
            self.assertIn(str(missing_repo.resolve()), str(context.exception))

    def test_vendored_prepare_plan_uses_internal_cli_entrypoint(self) -> None:
        downloader = LegacyMlebenchDownloader()
        plan = downloader.build_prepare_plan(
            "detecting-insults-in-social-commentary",
            Path("/tmp/cache"),
        )

        self.assertEqual(
            plan.command[1:4],
            ["-m", "aisci_domain_mle.vendored_lite_cli", "prepare"],
        )
        self.assertEqual(plan.module, "aisci_domain_mle.vendored_lite_cli")
        self.assertEqual(plan.cwd, str(Path(__file__).resolve().parents[2]))
        self.assertEqual(plan.env, {})

    def test_vendored_registry_rejects_non_lite_competitions(self) -> None:
        downloader = LegacyMlebenchDownloader()

        with self.assertRaises(ValueError) as context:
            downloader.build_prepare_plan("spaceship-titanic", Path("/tmp/cache"))

        self.assertIn("built-in MLE-bench Lite registry", str(context.exception))
        self.assertIn("spaceship-titanic", str(context.exception))

    def test_local_zip_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "detecting-insults-in-social-commentary.zip"
            zip_path.write_text("placeholder", encoding="utf-8")

            resolved = resolve_competition_source(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=str(zip_path),
                cache_root=tmp_path / "cache",
                allow_download=False,
            )

            self.assertEqual(resolved.source_kind, InputSourceKind.LOCAL_ZIP)
            self.assertEqual(resolved.competition_zip_path, str(zip_path.resolve()))
            self.assertTrue(resolved.zip_exists)
            self.assertIsNone(resolved.legacy_prepare_plan)

    def test_competition_name_uses_prepared_cache_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            prepared_dir = cache_root / "detecting-insults-in-social-commentary" / "prepared"
            prepared_dir.mkdir(parents=True)

            resolved = resolve_competition_source(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=None,
                cache_root=cache_root,
                allow_download=False,
            )

            self.assertEqual(resolved.source_kind, InputSourceKind.COMPETITION_NAME)
            self.assertTrue(resolved.cache_dir_exists)
            self.assertTrue(resolved.cache_prepared_exists)
            self.assertEqual(resolved.cache_prepared_dir, str(prepared_dir.resolve()))
            self.assertIsNone(resolved.legacy_prepare_plan)

    def test_cache_miss_returns_wired_legacy_prepare_plan_without_running_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "mle-bench"
            (repo_root / "mlebench").mkdir(parents=True)
            (repo_root / "mlebench" / "cli.py").write_text("print('placeholder')\n", encoding="utf-8")
            cache_root = tmp_path / "cache"
            calls: list[list[str]] = []

            def fake_runner(plan) -> None:
                calls.append(plan.command)

            downloader = LegacyMlebenchDownloader(repo_root=repo_root, runner=fake_runner)
            resolved = resolve_competition_source(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=None,
                cache_root=cache_root,
                allow_download=False,
                downloader=downloader,
            )

            self.assertFalse(calls)
            self.assertFalse(resolved.cache_dir_exists)
            self.assertFalse(resolved.cache_prepared_exists)
            self.assertIsNotNone(resolved.legacy_prepare_plan)
            assert resolved.legacy_prepare_plan is not None
            self.assertEqual(
                resolved.legacy_prepare_plan.command[1:4],
                ["-m", "mlebench.cli", "prepare"],
            )
            self.assertIn(str(repo_root.resolve()), resolved.legacy_prepare_plan.env["PYTHONPATH"])

    def test_cache_miss_can_delegate_to_legacy_prepare_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "mle-bench"
            (repo_root / "mlebench").mkdir(parents=True)
            (repo_root / "mlebench" / "cli.py").write_text("print('placeholder')\n", encoding="utf-8")
            cache_root = tmp_path / "cache"
            calls: list[str] = []

            def fake_runner(plan) -> None:
                calls.append(plan.competition_name)
                prepared_dir = Path(plan.data_dir) / plan.competition_name / "prepared"
                prepared_dir.mkdir(parents=True)

            downloader = LegacyMlebenchDownloader(repo_root=repo_root, runner=fake_runner)
            resolved = resolve_competition_source(
                competition_name="detecting-insults-in-social-commentary",
                competition_zip_path=None,
                cache_root=cache_root,
                allow_download=True,
                downloader=downloader,
            )

            self.assertEqual(calls, ["detecting-insults-in-social-commentary"])
            self.assertTrue(resolved.cache_dir_exists)
            self.assertTrue(resolved.cache_prepared_exists)
            self.assertIsNone(resolved.legacy_prepare_plan)

    def test_legacy_competition_preparer_runs_prepare_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "mle-bench"
            competition_dir = repo_root / "mlebench" / "competitions" / "demo-competition"
            competition_dir.mkdir(parents=True)
            (competition_dir / "config.yaml").write_text(
                "preparer: mlebench.competitions.demo-competition.prepare:prepare\n",
                encoding="utf-8",
            )
            (competition_dir / "prepare.py").write_text(
                "from pathlib import Path\n"
                "\n"
                "def prepare(raw: Path, public: Path, private: Path) -> None:\n"
                "    public.mkdir(parents=True, exist_ok=True)\n"
                "    private.mkdir(parents=True, exist_ok=True)\n"
                "    (public / 'train.csv').write_text((raw / 'train.csv').read_text(encoding='utf-8'), encoding='utf-8')\n"
                "    (private / 'answers.csv').write_text('secret\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )

            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            (raw_dir / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            public_dir = tmp_path / "prepared" / "public"
            private_dir = tmp_path / "prepared" / "private"

            preparer = LegacyCompetitionPreparer(repo_root=repo_root)
            preparer.prepare_local_dataset(
                "demo-competition",
                raw_dir=raw_dir,
                public_dir=public_dir,
                private_dir=private_dir,
            )

            self.assertEqual((public_dir / "train.csv").read_text(encoding="utf-8"), "x,y\n1,2\n")
            self.assertEqual((private_dir / "answers.csv").read_text(encoding="utf-8"), "secret\n")

    def test_legacy_competition_preparer_only_resolves_public_sample_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "mle-bench"
            competition_dir = repo_root / "mlebench" / "competitions" / "demo-competition"
            competition_dir.mkdir(parents=True)
            (competition_dir / "config.yaml").write_text(
                "preparer: mlebench.competitions.demo-competition.prepare:prepare\n"
                "description: mlebench/competitions/demo-competition/description.md\n"
                "dataset:\n"
                "  sample_submission: demo-competition/prepared/public/sample_submission.csv\n",
                encoding="utf-8",
            )
            (competition_dir / "description.md").write_text("# Demo\n", encoding="utf-8")

            prepared_dir = tmp_path / "staging" / "prepared"
            public_dir = prepared_dir / "public"
            private_dir = prepared_dir / "private"
            public_dir.mkdir(parents=True)
            private_dir.mkdir(parents=True)
            (public_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (private_dir / "sample_submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")

            preparer = LegacyCompetitionPreparer(repo_root=repo_root)
            description_path, sample_path = preparer.resolve_public_metadata_paths(
                "demo-competition",
                prepared_dir=prepared_dir,
            )

            self.assertEqual(description_path, (competition_dir / "description.md").resolve())
            self.assertEqual(sample_path, (public_dir / "sample_submission.csv").resolve())

    def test_legacy_competition_preparer_drops_private_sample_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "mle-bench"
            competition_dir = repo_root / "mlebench" / "competitions" / "demo-competition"
            competition_dir.mkdir(parents=True)
            (competition_dir / "config.yaml").write_text(
                "preparer: mlebench.competitions.demo-competition.prepare:prepare\n"
                "description: mlebench/competitions/demo-competition/description.md\n"
                "dataset:\n"
                "  sample_submission: demo-competition/prepared/private/sample_submission.csv\n",
                encoding="utf-8",
            )
            (competition_dir / "description.md").write_text("# Demo\n", encoding="utf-8")
            prepared_dir = tmp_path / "staging" / "prepared"
            (prepared_dir / "private").mkdir(parents=True)
            (prepared_dir / "private" / "sample_submission.csv").write_text(
                "id,target\n1,1\n",
                encoding="utf-8",
            )

            preparer = LegacyCompetitionPreparer(repo_root=repo_root)
            _, sample_path = preparer.resolve_public_metadata_paths(
                "demo-competition",
                prepared_dir=prepared_dir,
            )

            self.assertIsNone(sample_path)


if __name__ == "__main__":
    unittest.main()

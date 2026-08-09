import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.asr.engines import ASRFullResult, ASRSegmentResult
from app.services.speaker.domain import (
    SpeakerEmbedding,
    SpeakerMatchDecision,
    SpeakerMatchStatus,
    VoiceprintCandidate,
)
from app.services.speaker.matching import SpeakerMatcher
from app.services.speaker.stores.sqlite_vec_store import SqliteVecVoiceprintStore


class VoiceprintMatchingTest(unittest.TestCase):
    def test_matches_when_top_candidate_passes_threshold_and_margin(self) -> None:
        matcher = SpeakerMatcher(threshold=0.75, margin=0.08)

        decision = matcher.decide(
            local_speaker_id="Speaker1",
            candidates=[
                VoiceprintCandidate(
                    speaker_id="spk1",
                    display_name="Alice",
                    score=0.91,
                    max_score=0.93,
                    top3_mean_score=0.863333,
                    sample_count=3,
                ),
                VoiceprintCandidate(
                    speaker_id="spk2",
                    display_name="Bob",
                    score=0.72,
                    max_score=0.74,
                    top3_mean_score=0.673333,
                    sample_count=3,
                ),
            ],
        )

        self.assertEqual(decision.status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(decision.output_speaker_id, "Alice")

    def test_keeps_local_label_when_margin_is_too_small(self) -> None:
        matcher = SpeakerMatcher(threshold=0.75, margin=0.08)

        decision = matcher.decide(
            local_speaker_id="Speaker1",
            candidates=[
                VoiceprintCandidate(
                    speaker_id="spk1",
                    display_name="Alice",
                    score=0.91,
                    max_score=0.93,
                    top3_mean_score=0.863333,
                    sample_count=3,
                ),
                VoiceprintCandidate(
                    speaker_id="spk2",
                    display_name="Bob",
                    score=0.88,
                    max_score=0.9,
                    top3_mean_score=0.833333,
                    sample_count=3,
                ),
            ],
        )

        self.assertEqual(decision.status, SpeakerMatchStatus.UNKNOWN)
        self.assertEqual(decision.output_speaker_id, "Speaker1")

    def test_asr_segment_schema_is_not_extended(self) -> None:
        result = ASRFullResult(
            text="hello",
            duration=1.0,
            segments=[
                ASRSegmentResult(
                    text="hello",
                    start_time=0.0,
                    end_time=1.0,
                    speaker_id="Speaker1",
                )
            ],
        )

        segment = result.segments[0]
        self.assertEqual(segment.speaker_id, "Speaker1")
        self.assertFalse(hasattr(segment, "speaker_match"))
        self.assertFalse(hasattr(segment, "match_score"))
        self.assertFalse(hasattr(segment, "speaker_identity"))

    def test_sqlite_vec_speaker_score_uses_weighted_max_and_top3_mean(self) -> None:
        self.assertAlmostEqual(
            SqliteVecVoiceprintStore.calculate_speaker_score(
                max_score=0.8,
                top3_mean_score=0.5,
            ),
            0.71,
        )

    def test_sqlite_vec_store_groups_multiple_samples_by_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SqliteVecVoiceprintStore(
                db_path=str(Path(temp_dir) / "voiceprints.sqlite3")
            )
            store.ensure_schema()
            alice = store.create_speaker(
                display_name="Alice",
                description=None,
            )
            bob = store.create_speaker(
                display_name="Bob",
                description=None,
            )

            store.add_voiceprint(
                speaker_id=alice.id,
                embedding=self._embedding([1.0, 0.0, 0.0]),
                source_hash=None,
            )
            store.add_voiceprint(
                speaker_id=alice.id,
                embedding=self._embedding([0.95, 0.05, 0.0]),
                source_hash=None,
            )
            store.add_voiceprint(
                speaker_id=bob.id,
                embedding=self._embedding([0.0, 1.0, 0.0]),
                source_hash=None,
            )

            candidates = store.search(
                embedding=self._embedding([1.0, 0.0, 0.0]),
                limit=2,
            )

        self.assertEqual(candidates[0].display_name, "Alice")
        self.assertEqual(candidates[0].sample_count, 2)
        self.assertGreater(candidates[0].score, candidates[1].score)

    @staticmethod
    def _embedding(prefix: list[float]) -> SpeakerEmbedding:
        vector = np.zeros(192, dtype=np.float32)
        vector[: len(prefix)] = np.array(prefix, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return SpeakerEmbedding(
            vector=vector,
            provider="campplus",
            model_name="test-model",
            sample_rate=16000,
            duration_sec=3.0,
        )


class VoiceprintDeduplicationTest(unittest.TestCase):
    """测试 enric_asr_result 中的跨说话人声纹去重逻辑。"""

    def test_no_conflict_when_different_speakers_match_different_profiles(self) -> None:
        """两个本地说话人匹配到不同注册声纹 → 都保留 MATCHED。"""
        from app.services.speaker.identification_service import SpeakerIdentificationService

        decisions = {
            "说话人1": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人1",
                display_name="银行工作人员-男",
                score=0.91,
            ),
            "说话人2": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人2",
                display_name="银行工作人员-女",
                score=0.88,
            ),
        }
        result = SpeakerIdentificationService._deduplicate_decisions(decisions)

        self.assertEqual(result["说话人1"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人1"].output_speaker_id, "银行工作人员-男")
        self.assertEqual(result["说话人2"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人2"].output_speaker_id, "银行工作人员-女")

    def test_demotes_lower_score_when_two_speakers_match_same_profile(self) -> None:
        """两个本地说话人匹配到同一注册声纹 → 高分保留，低分降级 UNKNOWN。"""
        from app.services.speaker.identification_service import SpeakerIdentificationService

        decisions = {
            "说话人1": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人1",
                display_name="银行工作人员-男",
                score=0.91,
            ),
            "说话人3": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人3",
                display_name="银行工作人员-男",
                score=0.78,
            ),
        }
        result = SpeakerIdentificationService._deduplicate_decisions(decisions)

        # 说话人1 得分更高，保留
        self.assertEqual(result["说话人1"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人1"].output_speaker_id, "银行工作人员-男")
        # 说话人3 降级
        self.assertEqual(result["说话人3"].status, SpeakerMatchStatus.UNKNOWN)
        self.assertEqual(result["说话人3"].output_speaker_id, "说话人3")

    def test_demotes_all_but_best_when_three_speakers_match_same_profile(self) -> None:
        """三个本地说话人匹配到同一注册声纹 → 仅最高分 MATCHED。"""
        from app.services.speaker.identification_service import SpeakerIdentificationService

        decisions = {
            "说话人1": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人1",
                display_name="Alice",
                score=0.91,
            ),
            "说话人2": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人2",
                display_name="Alice",
                score=0.85,
            ),
            "说话人3": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人3",
                display_name="Alice",
                score=0.72,
            ),
        }
        result = SpeakerIdentificationService._deduplicate_decisions(decisions)

        self.assertEqual(result["说话人1"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人2"].status, SpeakerMatchStatus.UNKNOWN)
        self.assertEqual(result["说话人3"].status, SpeakerMatchStatus.UNKNOWN)

    def test_unknown_decisions_are_not_affected(self) -> None:
        """UNKNOWN 状态的决策不受去重影响。"""
        from app.services.speaker.identification_service import SpeakerIdentificationService

        decisions = {
            "说话人1": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人1",
                display_name="Alice",
                score=0.91,
            ),
            "说话人2": SpeakerMatchDecision(
                status=SpeakerMatchStatus.UNKNOWN,
                local_speaker_id="说话人2",
                score=0.65,
            ),
            "说话人3": SpeakerMatchDecision(
                status=SpeakerMatchStatus.UNKNOWN,
                local_speaker_id="说话人3",
                score=0.60,
            ),
        }
        result = SpeakerIdentificationService._deduplicate_decisions(decisions)

        self.assertEqual(result["说话人1"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人2"].status, SpeakerMatchStatus.UNKNOWN)
        self.assertEqual(result["说话人3"].status, SpeakerMatchStatus.UNKNOWN)

    def test_empty_and_single_decision_return_unchanged(self) -> None:
        """空决策和单决策直接返回原对象。"""
        from app.services.speaker.identification_service import SpeakerIdentificationService

        # 空
        result = SpeakerIdentificationService._deduplicate_decisions({})
        self.assertEqual(result, {})

        # 单个 MATCHED
        decisions = {
            "说话人1": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人1",
                display_name="Alice",
                score=0.91,
            ),
        }
        result = SpeakerIdentificationService._deduplicate_decisions(decisions)
        self.assertIs(result, decisions)  # 无冲突时原样返回

    def test_mixed_conflict_and_non_conflict(self) -> None:
        """混合场景：两个匹配到同一人 + 一个匹配到另一人 + 一个 UNKNOWN。"""
        from app.services.speaker.identification_service import SpeakerIdentificationService

        decisions = {
            "说话人1": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人1",
                display_name="银行工作人员-男",
                score=0.91,
            ),
            "说话人2": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人2",
                display_name="银行工作人员-女",
                score=0.88,
            ),
            "说话人3": SpeakerMatchDecision(
                status=SpeakerMatchStatus.MATCHED,
                local_speaker_id="说话人3",
                display_name="银行工作人员-男",
                score=0.78,
            ),
            "说话人4": SpeakerMatchDecision(
                status=SpeakerMatchStatus.UNKNOWN,
                local_speaker_id="说话人4",
                score=0.55,
            ),
        }
        result = SpeakerIdentificationService._deduplicate_decisions(decisions)

        # 说话人1 和 说话人3 冲突 → 说话人1 胜出
        self.assertEqual(result["说话人1"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人1"].output_speaker_id, "银行工作人员-男")
        self.assertEqual(result["说话人3"].status, SpeakerMatchStatus.UNKNOWN)
        # 说话人2 无冲突，保持
        self.assertEqual(result["说话人2"].status, SpeakerMatchStatus.MATCHED)
        self.assertEqual(result["说话人2"].output_speaker_id, "银行工作人员-女")
        # 说话人4 原本就是 UNKNOWN
        self.assertEqual(result["说话人4"].status, SpeakerMatchStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()

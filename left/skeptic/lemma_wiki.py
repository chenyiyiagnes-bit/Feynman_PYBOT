import json
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel


class VerifiedLemma(BaseModel):
    theorem_name: str
    domain_tag: str
    author_bot: str
    lean_snippet: str
    intuition_note: str
    verified_at: str


class FailureLesson(BaseModel):
    theorem_name: str
    author_bot: str
    compiler_error: str
    recorded_at: str


class LemmaWikiStore:
    """复现 DeepMind 跨智能体共享 Wiki 机制：已验证引理沉淀 + 全网失败教训广播"""
    def __init__(self, storage_path: str = "data/lemma_wiki.json"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save_data({"verified_lemmas": [], "failure_lessons": []})

    def _load_data(self) -> Dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save_data(self, data: Dict):
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def deposit_verified_lemma(self, lemma: VerifiedLemma):
        data = self._load_data()
        existing = [item for item in data["verified_lemmas"] if item["theorem_name"] != lemma.theorem_name]
        existing.append(lemma.model_dump())
        data["verified_lemmas"] = existing
        self._save_data(data)

    def record_failure_lesson(self, lesson: FailureLesson):
        data = self._load_data()
        data["failure_lessons"].append(lesson.model_dump())
        data["failure_lessons"] = data["failure_lessons"][-15:]
        self._save_data(data)

    def build_context_injection(self, domain_tag: str) -> tuple[str, str]:
        data = self._load_data()
        lemmas = [
            VerifiedLemma(**item) for item in data["verified_lemmas"]
            if item["domain_tag"] == domain_tag or domain_tag == "all"
        ]
        failures = [FailureLesson(**item) for item in data["failure_lessons"][-5:]]

        lean_preamble = "\n\n".join([
            f"-- [Wiki 已验证引理: {lm.theorem_name} | 贡献者: {lm.author_bot}]\n{lm.lean_snippet.strip()}"
            for lm in lemmas
        ])

        guide_lines = ["【共享引理 Wiki 可用工具箱（已通过编译器认证，可直接调用）】："]
        if not lemmas:
            guide_lines.append("   （当前暂无前置引理）")
        for lm in lemmas:
            guide_lines.append(f"   ✅ 定理 `{lm.theorem_name}`: {lm.intuition_note}")

        if failures:
            guide_lines.append("\n【全网避坑黑名单】：")
            for fl in failures:
                guide_lines.append(f"   ❌ 针对 `{fl.theorem_name}` 的失败教训: {fl.compiler_error}")

        return lean_preamble, "\n".join(guide_lines)

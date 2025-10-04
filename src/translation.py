"""
OPUS-MT Translation Module for Arabic Argument Mining.

This module handles the translation of English persuasive essays to Arabic
using the Helsinki-NLP OPUS-MT model, with alignment and label projection functionality.
"""

import ast
import os
import re
from typing import Any, Union, cast

from sklearn.metrics import classification_report
import torch
from simalign import SentenceAligner
from tqdm import tqdm
from transformers import TranslationPipeline, pipeline


class OPUSTranslator:
    """
    Handles translation of English text to Arabic using OPUS-MT with
    sentence-level translation per paragraph, alignment, and projection.
    """

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-tc-big-en-ar") -> None:
        """
        Initialize the OPUS-MT translator with aligner.

        Args:
            model_name: Name of the OPUS-MT model to use
        """
        self.model_name: str = model_name
        self.translator: TranslationPipeline = pipeline("translation", model=model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aligner: SentenceAligner = SentenceAligner(
            model="bert", token_type="bpe", matching_methods="i", device=device
        )

    def translate_text(self, text: str) -> str:
        """
        Translate a single text from English to Arabic.

        Args:
            text: English text to translate

        Returns:
            Arabic translation
        """
        if not text.strip():
            return ""

        result = self.translator(text)
        return result[0]["translation_text"]

    def read_bio_file(self, file_path: str) -> list[str]:
        """
        Read and parse sentences from a BIO annotation file at sentence-level per paragraph.

        Args:
            file_path: Path to the annotation file

        Returns:
            list of sentences (with empty strings for paragraph breaks)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines: list[str] = content.split("\n")
        tokens: list[str] = []

        # Extract tokens from BIO format
        for line in lines:
            if line.strip() == "":
                tokens.append("\n")
            else:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    tokens.append(parts[1].strip())

        # Group tokens into sentences
        sentences: list[str] = []
        start = 0

        for i in range(len(tokens)):
            if tokens[i] in [".", "?", "!"]:
                sentence = " ".join(tokens[start : i + 1])
                sentences.append(sentence)
                start = i + 1
            elif tokens[i] == "\n":
                if start < i:
                    sentence = " ".join(tokens[start:i])
                    if sentence.strip():
                        sentences.append(sentence)
                sentences.append("")
                start = i + 1

        # Add any remaining tokens
        if start < len(tokens):
            sentence = " ".join(tokens[start:])
            if sentence.strip():
                sentences.append(sentence)

        # Remove consecutive empty lines
        cleaned_sentences: list[str] = []
        prev_empty = False
        for sentence in sentences:
            if sentence == "":
                if not prev_empty:
                    cleaned_sentences.append("")
                prev_empty = True
            else:
                cleaned_sentences.append(sentence)
                prev_empty = False

        return cleaned_sentences

    def translate_sentences(self, sentences: list[str]) -> list[str]:
        """
        Translate a list of sentences from English to Arabic.

        Args:
            sentences: list of English sentences

        Returns:
            list of Arabic translations
        """
        translations: list[str] = []
        for sentence in tqdm(sentences, desc="Translating"):
            if not sentence.strip():
                translations.append("")
            else:
                translation = self.translate_text(sentence)
                translations.append(translation)
        return translations

    def translate_and_save(
        self,
        sentences: list[str],
        output_file: str,
    ) -> None:
        """
        Translate sentences and save to file in bilingual format.

        Args:
            sentences: list of English sentences to translate
            output_file: Path to save the translations
        """
        translations: list[str] = self.translate_sentences(sentences)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for orig, trans in zip(sentences, translations):
                if orig.strip() == "":
                    _ = f.write("\n")
                else:
                    _ = f.write(f"{orig}|||{trans}\n")

    def align_sentences(
        self,
        input_file: str,
        output_file_ar: str,
        output_file_en: str,
    ) -> None:
        """
        Create word alignments between English and Arabic sentences.

        Args:
            input_file: Path to the bilingual translation file
            output_file_ar: Path to save Arabic alignments
            output_file_en: Path to save English alignments
            target_lang: Target language for alignment ('ar' or 'en')
        """
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Process for Arabic alignments
        align_ar: list[str | list[tuple[int, int]]] = []
        for line in tqdm(text.split("\n"), desc="Aligning (EN->AR)"):
            if line == "":
                align_ar.append("")
                continue

            if "|||" not in line:
                continue

            sentence_pair = line.split("|||")
            if len(sentence_pair) < 2:
                continue

            # Normalize Arabic punctuation
            sentence_pair[1] = re.sub("؟", "?", sentence_pair[1])
            sentence_pair[1] = re.sub("،", ",", sentence_pair[1])
            sentence_pair[1] = re.sub("!", "!", sentence_pair[1])

            source_words = sentence_pair[0].split(" ")
            target_words = sentence_pair[1].split(" ")

            alignments_ar: dict[str, list[tuple[int, int]]] = (
                self.aligner.get_word_aligns(source_words, target_words)
            )
            align_ar.append(alignments_ar["itermax"])

        # Save Arabic alignments
        os.makedirs(os.path.dirname(output_file_ar), exist_ok=True)
        with open(output_file_ar, "w", encoding="utf-8") as f:
            for alignment in align_ar:
                _ = f.write(str(alignment) + "\n")

        # Process for English alignments (reverse direction)
        align_en: list[str | list[tuple[int, int]]] = []
        for line in tqdm(text.split("\n"), desc="Aligning (AR->EN)"):
            if line == "":
                align_en.append("")
                continue

            if "|||" not in line:
                continue

            sentence_pair = line.split("|||")
            if len(sentence_pair) < 2:
                continue

            # Normalize Arabic punctuation
            sentence_pair[1] = re.sub("؟", "?", sentence_pair[1])
            sentence_pair[1] = re.sub("،", ",", sentence_pair[1])
            sentence_pair[1] = re.sub("!", "!", sentence_pair[1])

            source_words = sentence_pair[1].split(" ")  # Reversed
            target_words = sentence_pair[0].split(" ")  # Reversed

            alignments_en: dict[str, list[tuple[int, int]]] = (
                self.aligner.get_word_aligns(source_words, target_words)
            )
            align_en.append(alignments_en["itermax"])

        # Save English alignments
        os.makedirs(os.path.dirname(output_file_en), exist_ok=True)
        with open(output_file_en, "w", encoding="utf-8") as f:
            for alignment in align_en:
                _ = f.write(str(alignment) + "\n")

    def read_annotations(self, file_path: str) -> list[list[list]]:
        """
        Read BIO annotations from file.

        Args:
            file_path: Path to the annotation file

        Returns:
            list of sentences, each containing [index, word, label] triplets
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().split("\n")

        # Parse into list of [index, word, label]
        original = []
        for line in content:
            parts = line.split("\t")
            original.append(parts)

        # Group by sentences (separated by empty lines)
        annotations = []
        start = 0
        for i in range(len(original)):
            if len(original[i]) < 2:
                if start < i:
                    annotations.append(original[start:i])
                start = i + 1
            elif i == len(original) - 1:
                annotations.append(original[start : i + 1])

        # Reindex each sentence starting from 0
        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                annotations[i][j][0] = j

        return annotations

    def create_ranges(self, annotations: list[list[list]]) -> list[list[list]]:
        """
        Create ranges of argument components from BIO annotations.

        Args:
            annotations: list of annotated sentences

        Returns:
            list of ranges [label, start_idx, end_idx] for each sentence
        """
        ranges = []
        for i in range(len(annotations)):
            j = 0
            k = 0
            ranges.append([])
            while j < len(annotations[i]):
                if annotations[i][j][2] != "O":
                    ranges[i].append([])
                    ranges[i][k].append(annotations[i][j][2])
                    ranges[i][k].append(annotations[i][j][0])
                    for l in range(j + 1, len(annotations[i])):
                        if annotations[i][l][2] == "O" or "B" in annotations[i][l][2]:
                            ranges[i][k].append(l)
                            j = l
                            k += 1
                            break
                    else:
                        # Reached end of sentence
                        ranges[i][k].append(len(annotations[i]))
                        j = len(annotations[i])
                        k += 1
                else:
                    j += 1

        return ranges

    def extract_sentences(self, text: str) -> list[str]:
        """
        Extract sentences from bilingual text.
        """
        return text.split("|||")

    def find_sentence(
        self, idx: int, paragraph: str, lang: str = "ar"
    ) -> tuple[int, int]:
        """
        Find which sentence a word belongs to and its position within that sentence.

        Args:
            idx: Global word index in paragraph
            paragraph: The paragraph text (bilingual format)
            lang: Language to use ('ar' for Arabic/target, 'en' for English/source)

        Returns:
            tuple of (sentence_index, position_in_sentence)
        """
        lines = paragraph.split("\n")
        while idx >= 0:
            for i in range(len(lines)):
                if "|||" not in lines[i]:
                    continue
                if lang == "ar":
                    sline = self.extract_sentences(lines[i])[0]
                else:
                    sline = self.extract_sentences(lines[i])[1]
                words = sline.split(" ")
                if idx >= len(words):
                    idx -= len(words)
                else:
                    return i, idx
        return 0, 0

    def find_min(self, alignment: list[tuple[int, int]], start: int, stop: int) -> int:
        """
        Find the minimum aligned position for a range.

        Args:
            alignment: list of (source, target) alignment pairs
            start: Start position in source
            stop: Stop position in source

        Returns:
            Minimum position in target
        """
        min_val = 100000
        for i in range(len(alignment)):
            if (
                alignment[i][0] >= start
                and alignment[i][0] <= stop
                and alignment[i][1] < min_val
            ):
                min_val = alignment[i][1]
        return 0 if min_val == 100000 else min_val

    def find_max(self, alignment: list[tuple[int, int]], start: int, stop: int) -> int:
        """
        Find the maximum aligned position for a range.

        Args:
            alignment: list of (source, target) alignment pairs
            start: Start position in source
            stop: Stop position in source

        Returns:
            Maximum position in target
        """
        max_val = 0
        for i in range(len(alignment)):
            if (
                alignment[i][0] >= start
                and alignment[i][0] <= stop
                and alignment[i][1] > max_val
            ):
                max_val = alignment[i][1]
        return max_val

    def project_annotations(
        self,
        source_file: str,
        translation_file: str,
        alignment_file: str,
        output_file: str,
        target_lang: str = "ar",
    ) -> None:
        """
        Project BIO annotations from source to target language.

        Args:
            source_file: Path to source BIO annotations
            translation_file: Path to bilingual translation file
            alignment_file: Path to alignment file
            output_file: Path to save projected annotations
            target_lang: Target language ('ar' or 'en')
        """
        # Read source annotations
        annotations = self.read_annotations(source_file)
        ranges = self.create_ranges(annotations)

        # Read translations
        with open(translation_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Read alignments
        with open(alignment_file, "r", encoding="utf-8") as f:
            alignments_text = f.read().split("\n\n")

        def parse_alignments(alignments):
            for i in range(len(alignments)):
                if len(alignments[i]) < 2:
                    alignments[i] = []
                    continue
                try:
                    alignments[i] = eval(alignments[i])
                except:
                    alignments[i] = []
            return alignments

        # Find target ranges using alignments
        target_ranges = []
        for i in range(len(ranges)):
            if len(ranges[i]) == 0:
                target_ranges.append([])
            else:
                target_ranges.append([])
                # Extract paragraph from text
                paragraphs = text.split("\n\n")
                if i >= len(paragraphs):
                    continue
                paragraph = paragraphs[i]

                for j in range(len(ranges[i])):
                    sentence, start = self.find_sentence(
                        ranges[i][j][1], paragraph, target_lang
                    )
                    sentence, stop = self.find_sentence(
                        ranges[i][j][2], paragraph, target_lang
                    )

                    # Parse alignment for this sentence
                    if i < len(alignments_text):
                        alignment = parse_alignments(alignments_text[i].split("\n"))
                        if (
                            sentence < len(alignment)
                            and alignment[sentence]
                            and isinstance(alignment[sentence], list)
                        ):
                            min_pos = self.find_min(alignment[sentence], start, stop)
                            max_pos = self.find_max(alignment[sentence], start, stop)
                            target_ranges[i].append(
                                [ranges[i][j][0], min_pos, max_pos, sentence]
                            )

        # Extract target text
        target_text = []
        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            target_text.append([])
            if "|||" not in paragraph:
                continue
            for line in paragraph.split("\n"):
                if "|||" not in line:
                    continue
                if target_lang == "en":
                    target_text[-1].append(self.extract_sentences(line)[0])
                else:
                    if len(self.extract_sentences(line)) > 1:
                        target_text[-1].append(self.extract_sentences(line)[1])

        # Initialize mask
        mask = []
        for i in range(len(target_text)):
            mask.append([])
            for j in range(len(target_text[i])):
                mask[i].append([])
                target_words = target_text[i][j].split(" ")
                for k in range(len(target_words)):
                    mask[i][j].append("O")

        for i in range(len(target_ranges)):
            for j in range(len(target_ranges[i])):
                for k in range(target_ranges[i][j][1], target_ranges[i][j][2]):
                    mask[i][target_ranges[i][j][3]][k] = target_ranges[i][j][0]

        # Convert mask to BIO format
        out = []
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if j >= len(target_text[i]):
                    continue
                target_words = target_text[i][j].split(" ")
                for k in range(len(mask[i][j])):
                    if k >= len(target_words):
                        continue
                    word = target_words[k]
                    label = mask[i][j][k]

                    if mask[i][j][k] == "O":
                        out.append([target_words[k], "O"])
                    elif mask[i][j][k] != mask[i][j][k - 1]:
                        out.append([target_words[k], "B" + mask[i][j][k][1:]])
                    else:
                        out.append([target_words[k], "I" + mask[i][j][k][1:]])
            out.append([])

        # Remove trailing empty lines
        while out and out[-1] == []:
            out.pop()

        # Format output
        output = ""
        count = 1
        for i in range(len(out)):
            if len(out[i]) == 0:
                output += "\n"
                count = 1
                continue
            if i == len(out) - 1:
                output += f"{count}\t{out[i][0]}\t{out[i][1]}"
            else:
                output += f"{count}\t{out[i][0]}\t{out[i][1]}\n"
                count += 1

        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)


def translate_dataset(
    input_dir: str, output_dir: str, splits: list[str] | None = None
) -> None:
    """
    Translate all dataset splits using OPUS-MT with sentence-level translation.

    Args:
        input_dir: Directory containing original BIO files
        output_dir: Directory to save translated files
        splits: Dataset splits to translate
    """
    if splits is None:
        splits = ["train", "dev", "test"]

    translator = OPUSTranslator()

    for split in splits:
        input_file = os.path.join(input_dir, f"{split}.dat.abs")
        output_file = os.path.join(output_dir, f"{split}-OPUS.txt")

        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {split} split...")
        print(f"{'=' * 60}")

        # Step 1: Translation
        print("\n[1/3] Translating...")
        sentences = translator.read_bio_file(input_file)
        translator.translate_and_save(sentences, output_file)
        print(f"✓ Saved translations to {output_file}")

        # Step 2: Alignment
        print("\n[2/3] Creating alignments...")
        align_file_ar = os.path.join(output_dir, f"{split}-OPUS-alignar.txt")
        align_file_en = os.path.join(output_dir, f"{split}-OPUS-alignen.txt")
        translator.align_sentences(output_file, align_file_ar, align_file_en)
        print(f"✓ Saved alignments to {align_file_ar} and {align_file_en}")

        # Step 3: Projection
        print("\n[3/3] Projecting annotations...")
        proj_file_ar = os.path.join(output_dir, f"{split}-OPUS-annotationsar.txt")
        proj_file_en = os.path.join(output_dir, f"{split}-OPUS-annotationsen.txt")

        translator.project_annotations(
            input_file, output_file, align_file_ar, proj_file_ar, "ar"
        )
        translator.project_annotations(
            proj_file_ar, output_file, align_file_en, proj_file_en, "en"
        )
        print(f"✓ Saved projections to {proj_file_ar} and {proj_file_en}")

        print(f"\n✓ Completed {split} split")


def evaluate():
    target_classes = [
        "O",
        "B-MajorClaim",
        "I-MajorClaim",
        "B-Claim",
        "I-Claim",
        "B-Premise",
        "I-Premise",
    ]
    gold_standard_train = open("./data/raw/train.dat.abs", "r").read().split("\n")
    gold_standard_dev = open("./data/raw/dev.dat.abs", "r").read().split("\n")
    gold_standard_test = open("./data/raw/test.dat.abs", "r").read().split("\n")

    while len(gold_standard_train[-1]) < 2:
        gold_standard_train.pop()
    while len(gold_standard_dev[-1]) < 2:
        gold_standard_dev.pop()
    while len(gold_standard_test[-1]) < 2:
        gold_standard_test.pop()

    gold_standard = gold_standard_train + gold_standard_dev + gold_standard_test

    projected_train = (
        open("./data/translated/train-OPUS-annotationsen.txt", "r").read().split("\n")
    )
    projected_dev = (
        open("./data/translated/dev-OPUS-annotationsen.txt", "r").read().split("\n")
    )
    projected_test = (
        open("./data/translated/test-OPUS-annotationsen.txt", "r").read().split("\n")
    )

    projected = projected_train + projected_dev + projected_test

    i = 0

    while i < (len(gold_standard)):
        gold_standard[i] = gold_standard[i].split("\t")
        projected[i] = projected[i].split("\t")

        if len(gold_standard[i]) != 3:
            gold_standard.pop(i)
            projected.pop(i)
            continue

        gold_standard[i] = gold_standard[i][2]
        projected[i] = projected[i][2]

        if ":" in gold_standard[i]:
            index = gold_standard[i].index(":")
            gold_standard[i] = gold_standard[i][0:index]

        if ":" in projected[i]:
            index = projected[i].index(":")
            projected[i] = projected[i][0:index]

        if gold_standard[i] not in target_classes:
            print(gold_standard[i], len(gold_standard[i]))

        i += 1

    print(
        classification_report(
            gold_standard, projected, target_names=target_classes, digits=4
        )
    )


if __name__ == "__main__":
    evaluate()

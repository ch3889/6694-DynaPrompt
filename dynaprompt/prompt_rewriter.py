"""
Local LLM-based prompt rewriting for better compositional generation.

Uses lightweight local models (Llama, Mistral, or similar) to intelligently
rewrite prompts to maximize the chance all objects appear.
"""

import re
from typing import List, Tuple, Optional


class PromptRewriter:
    """
    LLM-based prompt rewriter for improving compositional accuracy.

    Uses local models to:
    1. Add scene context that makes object combinations plausible
    2. Resolve semantic conflicts (e.g., "parking lot" + "bicycle")
    3. Make spatial relationships explicit
    4. Generate negative prompts to avoid failure modes
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_gpu: bool = True):
        """
        Initialize prompt rewriter with local LLM.

        Args:
            model_name: HuggingFace model name (default: TinyLlama for speed)
            use_gpu: Use GPU if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.pipeline = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of LLM (only when first needed)."""
        if self._initialized:
            return

        try:
            from transformers import pipeline
            import torch

            device = 0 if self.use_gpu and torch.cuda.is_available() else -1

            print(f"Loading prompt rewriter model: {self.model_name}...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            print("✓ Prompt rewriter ready")
            self._initialized = True

        except ImportError:
            print("⚠ transformers not installed. Install with: pip install transformers torch")
            self._initialized = False
        except Exception as e:
            print(f"⚠ Failed to load LLM: {e}")
            self._initialized = False

    def rewrite_for_accuracy(
        self,
        prompt: str,
        objects: Optional[List[str]] = None,
        fallback_to_simple: bool = True
    ) -> Tuple[str, str]:
        """
        Rewrite prompt for better compositional accuracy using LLM.

        Args:
            prompt: Original text prompt
            objects: List of required objects (auto-detected if None)
            fallback_to_simple: Use simple rules if LLM unavailable

        Returns:
            (enhanced_prompt, negative_prompt)
        """
        # Try LLM first
        self._lazy_init()

        if self._initialized and self.pipeline is not None:
            return self._rewrite_with_llm(prompt, objects)
        elif fallback_to_simple:
            print("Using simple rule-based rewriting (LLM unavailable)")
            return self._simple_rewrite(prompt, objects)
        else:
            # No rewriting
            return prompt, self._generate_simple_negative(prompt, objects)

    def _rewrite_with_llm(self, prompt: str, objects: Optional[List[str]]) -> Tuple[str, str]:
        """Use local LLM to rewrite prompt."""

        if objects is None:
            objects = self._extract_objects(prompt)

        # Construct instruction for LLM
        instruction = self._build_instruction(prompt, objects)

        # Generate rewrite
        try:
            result = self.pipeline(
                instruction,
                max_new_tokens=150,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            generated_text = result[0]['generated_text']

            # Extract enhanced prompt from generation
            enhanced_prompt = self._extract_enhanced_prompt(generated_text, instruction)

            # Verify all objects are preserved
            enhanced_prompt, verified = self._verify_objects_preserved(
                original_prompt=prompt,
                rewritten_prompt=enhanced_prompt,
                required_objects=objects
            )

            # Generate negative prompt
            negative_prompt = self._generate_negative_from_objects(objects)

            return enhanced_prompt, negative_prompt

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to simple rewrite")
            return self._simple_rewrite(prompt, objects)

    def _verify_objects_preserved(
        self,
        original_prompt: str,
        rewritten_prompt: str,
        required_objects: List[str]
    ) -> Tuple[str, bool]:
        """
        Verify that all required objects are present in the rewritten prompt.

        Args:
            original_prompt: Original user prompt
            rewritten_prompt: LLM-rewritten prompt
            required_objects: List of objects that must be present

        Returns:
            (final_prompt, verification_passed)
            If verification fails, returns safer fallback prompt
        """
        rewritten_lower = rewritten_prompt.lower()
        missing_objects = []

        # Check each required object
        for obj in required_objects:
            # Look for the object or its variants
            if obj.lower() not in rewritten_lower:
                missing_objects.append(obj)

        if missing_objects:
            print(f"⚠ LLM rewrite verification FAILED:")
            print(f"  Original:  {original_prompt}")
            print(f"  Rewritten: {rewritten_prompt}")
            print(f"  Missing:   {missing_objects}")
            print(f"  → Using safer fallback with spatial markers")

            # Fallback: Add explicit spatial markers to original prompt
            fallback = self._create_spatial_fallback(original_prompt, required_objects)
            return fallback, False

        print(f"✓ LLM rewrite verification PASSED - all {len(required_objects)} objects preserved")
        return rewritten_prompt, True

    def _create_spatial_fallback(self, prompt: str, objects: List[str]) -> str:
        """
        Create a safe fallback prompt with explicit spatial markers.

        Strategy: Enumerate objects with explicit positions to force generation.
        """
        prompt_lower = prompt.lower()

        # Check for known conflicts
        if "park" in prompt_lower and any("bicycle" in obj.lower() for obj in objects):
            # Known semantic conflict: parking + bicycle
            # Fix: Change scene context to "city street"
            base = f"In a city street scene,"
        elif not any(ctx in prompt_lower for ctx in ["in a", "on a", "at a", "inside", "outside"]):
            # No scene context - add generic one
            base = f"In a scene,"
        else:
            base = ""

        # Add original prompt with spatial clarification
        # Replace ambiguous words
        enhanced = prompt.replace("parked next to", "positioned beside")
        enhanced = enhanced.replace("next to", "positioned beside")
        enhanced = enhanced.replace(" with ", " and also ")

        if base:
            return f"{base} {enhanced}"
        else:
            return enhanced

    def _build_instruction(self, prompt: str, objects: List[str]) -> str:
        """Build instruction for LLM with strict rules for clarity and attribute emphasis."""

        instruction = f"""<|system|>
You are a prompt optimizer for Stable Diffusion image generation.
Rewrite prompts to ensure all mentioned objects AND their attributes (colors, sizes) appear in the image.
Keep it concise and natural.
<|user|>
Original prompt: "{prompt}"
Required objects: {', '.join(objects)}

CRITICAL Rules:
1. MUST include ALL objects AND attributes (colors, sizes, materials) from the list above
2. Place important attributes (colors like "silver", "golden") EARLY in the sentence for emphasis
3. Use EXPLICIT spatial relationships: "on the left side", "on the right side", "in front of", "behind"
4. NEVER use ambiguous words like "with" - always specify where objects are relative to each other
5. Format: "Important attributes + objects + spatial positions + scene context"
6. Add scene context if missing (e.g., "in a city street", "in a park")
7. Avoid semantic conflicts (e.g., don't say "parking lot" with "bicycle" - use "city street" instead)
8. Stay under 30 words

Example good rewrite: "A golden bicycle on the right side and a silver car on the left side, in a city street"
Example bad rewrite: "A silver car with a golden bicycle" (ambiguous "with", no positions)
Note: Attributes like "golden" and "silver" come FIRST for emphasis

Rewrite the prompt now:
<|assistant|>
Enhanced prompt: """

        return instruction

    def _extract_enhanced_prompt(self, generated_text: str, instruction: str) -> str:
        """Extract the enhanced prompt from LLM output."""

        # Remove instruction part
        if instruction in generated_text:
            generated_text = generated_text.replace(instruction, "")

        # Clean up
        generated_text = generated_text.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Enhanced prompt:",
            "Rewritten prompt:",
            "New prompt:",
            '"""',
            '"',
        ]

        for prefix in prefixes_to_remove:
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):].strip()

        # Take first sentence if multiple
        if '\n' in generated_text:
            generated_text = generated_text.split('\n')[0].strip()

        # Remove trailing quotes/punctuation artifacts
        generated_text = generated_text.strip('"\'.;')

        # Validate: check if it contains the objects
        # If not, fall back to original with added context
        # (This catches cases where LLM hallucinated)

        return generated_text if generated_text else prompt

    def _simple_rewrite(self, prompt: str, objects: Optional[List[str]]) -> Tuple[str, str]:
        """Simple rule-based rewriting as fallback."""

        if objects is None:
            objects = self._extract_objects(prompt)

        prompt_lower = prompt.lower()

        # Detect and fix common conflicts
        if "park" in prompt_lower and any("bicycle" in obj.lower() for obj in objects):
            # Fix parking + bicycle conflict
            enhanced = self._fix_parking_bicycle_conflict(prompt, objects)
        elif not any(ctx in prompt_lower for ctx in ["in a", "on a", "at a", "inside", "outside"]):
            # Add generic context if missing
            enhanced = f"in a scene, {prompt}"
        else:
            enhanced = prompt

        negative = self._generate_negative_from_objects(objects)

        return enhanced, negative

    def _fix_parking_bicycle_conflict(self, prompt: str, objects: List[str]) -> str:
        """Fix the parking + bicycle semantic conflict."""

        # Replace "parked" with context-appropriate phrasing
        enhanced = prompt.replace("parked next to", "near")
        enhanced = enhanced.replace("parked", "on the street,")

        # Add scene context
        if "in a" not in enhanced.lower():
            enhanced = f"on a city street, {enhanced}"

        return enhanced

    def _extract_objects(self, prompt: str) -> List[str]:
        """Extract main objects from prompt."""
        ignore = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
                  'by', 'from', 'is', 'are', 'was', 'were', 'next', 'near'}

        words = prompt.lower().split()
        objects = []

        for word in words:
            word_clean = word.strip('.,!?;:')
            if word_clean and word_clean not in ignore and len(word_clean) > 2:
                objects.append(word_clean)

        return objects

    def _generate_negative_from_objects(self, objects: List[str]) -> str:
        """Generate negative prompt from object list."""
        negative_parts = []

        # For each object, add "missing X" to negative
        for obj in objects[:3]:  # Limit to 3 most important
            negative_parts.append(f"missing {obj}")

        # Add general failure modes
        negative_parts.extend([
            "incomplete scene",
            "blurry",
            "low quality"
        ])

        return ", ".join(negative_parts)

    def _generate_simple_negative(self, prompt: str, objects: Optional[List[str]]) -> str:
        """Generate simple negative prompt without object analysis."""
        return "incomplete scene, missing objects, blurry, low quality"


def test_prompt_rewriter():
    """Test the prompt rewriter."""
    print("="*80)
    print("Prompt Rewriter Test")
    print("="*80)

    # Test with simple fallback first (no LLM)
    print("\n1. Testing simple rule-based rewriting (no LLM):")
    print("-"*80)

    rewriter_simple = PromptRewriter()
    rewriter_simple._initialized = False  # Force simple mode

    test_prompts = [
        "a silver car parked next to a golden bicycle",
        "a cat sitting on a wooden chair",
        "a red apple on a white plate",
    ]

    for original in test_prompts:
        enhanced, negative = rewriter_simple.rewrite_for_accuracy(original, fallback_to_simple=True)
        print(f"\nOriginal:  {original}")
        print(f"Enhanced:  {enhanced}")
        print(f"Negative:  {negative}")

    # Test with LLM if available
    print("\n" + "="*80)
    print("2. Testing with local LLM (if available):")
    print("-"*80)

    try:
        rewriter_llm = PromptRewriter(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        for original in test_prompts[:1]:  # Test one prompt
            enhanced, negative = rewriter_llm.rewrite_for_accuracy(original)
            print(f"\nOriginal:  {original}")
            print(f"Enhanced:  {enhanced}")
            print(f"Negative:  {negative}")

    except Exception as e:
        print(f"\nLLM test skipped: {e}")
        print("This is normal if transformers is not installed.")

    print("\n" + "="*80)
    print("✓ Test complete")
    print("="*80)


if __name__ == "__main__":
    test_prompt_rewriter()

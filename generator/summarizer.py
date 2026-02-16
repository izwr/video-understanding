from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class VideoSummarizer:
    def __init__(self, model_name: str = "DAMO-NLP-SG/VideoLLaMA3-7B") -> None:
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    @torch.no_grad()
    def summarize(self, video_path: str, context: str) -> str:
        """Generate a video summary using the video and structured context.

        Args:
            video_path: Path to the video file.
            context: Structured text describing detected objects and actions.

        Returns:
            Summary string from the model.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path}},
                    {
                        "type": "text",
                        "text": (
                            "You are a video analysis assistant. Using the video and the "
                            "following detection context, provide a detailed summary of "
                            "what is happening in the video.\n\n"
                            f"Detection Context:\n{context}\n\n"
                            "Provide a comprehensive summary covering: the setting/scene, "
                            "the people and their actions, key objects present, and the "
                            "overall narrative of what is happening."
                        ),
                    },
                ],
            }
        ]

        inputs = self.processor(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        # Strip the input tokens from output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        summary = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return summary.strip()

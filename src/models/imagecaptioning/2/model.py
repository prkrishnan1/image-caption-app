# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class TritonPythonModel:
    def initialize(self, args):
        self.processor = AutoProcessor.from_pretrained(
            "/workspace/imagecaptionapp/src/tokenizers/imagecaptioning/2/"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "/workspace/imagecaptionapp/src/models/imagecaptioning/2/"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            input_numpy = np.squeeze(inp.as_numpy()) 
            
            input_image = Image.fromarray(input_numpy, mode='RGB')
            inputs = self.processor(images=input_image, return_tensors="pt")

            generated_ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(generated_caption)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "text_output", np.array([generated_caption]).astype(pb_utils.TRITON_STRING_TO_NUMPY["TYPE_STRING"])
                    )
                ]
            )
            
            responses.append(inference_response)
        return responses
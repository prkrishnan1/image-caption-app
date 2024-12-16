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
import logging
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *
# from transformers import AutoProcessor, AutoModelForCausalLM

logging.basicConfig(filename='tritonclient.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

class TritonInferenceServer:
    def __init__(self, model_name="imagecaptioning", port="8000"):
        self.client = httpclient.InferenceServerClient(url="localhost:{}".format(port))
        self.model_name = model_name
    
    def predict(self, local_image_path: str) -> str:
        """
        Given a local image path, return caption from model hosted on Triton Inference Server
        """
        image = np.asarray(Image.open(local_image_path)).astype(np.uint8) # Bug fix: cast input image array as np.uint8 because PIL.Image.fromarray does not process np.float32 correctly

        # Set Inputs
        image = np.expand_dims(image, axis=0)
        input_tensors = [httpclient.InferInput("image", shape=image.shape, datatype="UINT8")] 
        logger.info("Image Shape: {}".format(image.shape))
        input_tensors[0].set_data_from_numpy(image)

        # Set outputs
        outputs = [httpclient.InferRequestedOutput("text_output")]

        # Query
        query_response = self.client.infer(
            model_name=self.model_name, inputs=input_tensors, outputs=outputs
        )

        # Output
        generated_caption = query_response.as_numpy("text_output")[0].decode("utf-8")
        logger.info("Triton Inference Response Using Trained Model: ")
        logger.info(generated_caption)

        return generated_caption

    def predict_local(self, local_image_path: str) -> str:
        """
        Given a local image path, return caption from model initialized locally
        """

        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        
        input_image = Image.open(local_image_path) 

        inputs = processor(images=input_image, return_tensors="pt")

        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info("Local Base Model Inference Response: ")
        logger.info(generated_caption)

        return generated_caption

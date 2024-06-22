### Setting Up the GPT Environment

This README provides a structured approach to setting up a robust environment for working with the GPT-2 model, ensuring compatibility and proper configuration of all dependencies.

#### 1. Create and Activate Conda Environment
First, create a new conda environment named `GPT`.

```sh
conda create --name GPT
conda activate GPT
```

#### 2. Install Required Libraries
Install PyTorch, torchvision, torchaudio, and the necessary CUDA dependencies. Additionally, install the `transformers` library from Hugging Face and the Intel Math Kernel Library (MKL).

```sh
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers
pip install tiktoken
```

#### 3. Install Specific Version of NumPy
Uninstall the existing version of NumPy and install NumPy version 1.26.4.

```sh
pip uninstall numpy
pip install numpy==1.26.4
```

### References
For more information on the GPT-2 and GPT-3 models, you can refer to the following papers:

- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Note:** There is a known discrepancy in the GPT-2 paper regarding the calculation of model parameters. The 117M parameters are supposed to be 124M as discussed in the [GPT-2 GitHub repository](https://github.com/openai/gpt-2?tab=readme-ov-file#gpt-2).

### Project Aim
The aim of this project is to surpass the performance of the GPT-2 124M model. The original GPT-2 code was implemented in TensorFlow, but this project will be implemented in PyTorch.

### Verifying the GPT-2 Implementation

To verify the GPT-2 implementation, you can use the following script to load the pre-trained weights and ensure the model behaves as expected.

1. **Navigate to the Test Implementation Directory**

```sh
cd test_implementation
```

2. **Run the Test Script**

```sh
python "gpt2 implementation test.py"
```

This script will load the GPT-2 model with the pre-trained weights and perform a simple test to verify that the model is functioning correctly.

**Verification Message:**
If the terminal outputs "Did not crash," then you have successfully verified the implementation.

By following these steps, you will ensure that your environment is set up correctly and that the GPT-2 implementation is verified and ready for further development.
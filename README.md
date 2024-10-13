<div align="center">

</div>

## QuizWiz

Outspeed is a PyTorch-inspired SDK for building real-time AI applications on voice and video input. It offers:

- Low-latency processing of streaming audio and video
- Intuitive API familiar to PyTorch users
- Flexible integration of custom AI models
- Tools for data preprocessing and model deployment

Ideal for developing voice assistants, video analytics, and other real-time AI applications processing audio-visual data.

## Install

You can install `outspeed` SDK from pypi using

```
pip install "outspeed[silero]>=0.1.143"
```

This would install the core `outspeed` package.
Read our [quickstart guide](https://docs.outspeed.com/examples/quickstart) to get started.

### Usage

Read the [docs](http://docs.outspeed.com) to learn more about the SDK.

To deploy your realtime function on Outspeed's infra, you can use the `outspeed deploy` CLI.

```
# functions.py contains your realtime function code
outspeed deploy --api-key=<your-api-key> functions.py
```

[Contact us](mailto:contact@outspeed.com) to get an API key and deploy.

Once deployed, you can use the playground in the examples repo to test the deployed code.

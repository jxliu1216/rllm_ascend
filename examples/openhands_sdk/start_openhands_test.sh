docker run --rm -it \
  -e LLM_BASE_URL="https://api.kimi.com/coding/" \
  -e LLM_API_KEY="sk-kimi-VdAI9VWur5mGEE801IWQZlJ8YqzUN1BS6QNRv0a9A1CTXijkCBSBfrTshFop3cmL" \
  -e LLM_MODEL="openai/Kimi-K2.5" \
  -e WORKSPACE_BASE=/opt/workspace/agent_workdir \
  -e MAX_ITERATIONS=10 \
  -e OBSERVER_API_URL="http://host.docker.internal:18858" \
  -e OBSERVER_SESSION_ID="my-rollout-001" \
  -e OBSERVER_SESSION_LABEL="softmax-operator-test" \
  -v /home/l00619433/rllm_ascend/examples/openhands_sdk/workspace:/opt/workspace \
  --network host \
  --add-host host.docker.internal:host-gateway \
  --entrypoint /opt/workspace/entrypoint.py \
  openhands-triton-env:v1
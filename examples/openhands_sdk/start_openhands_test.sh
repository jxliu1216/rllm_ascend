docker run --rm -it \
  -e LLM_BASE_URL="https://api.kimi.com/coding/v1" \
  -e LLM_API_KEY="sk-kimi-VdAI9VWur5mGEE801IWQZlJ8YqzUN1BS6QNRv0a9A1CTXijkCBSBfrTshFop3cmL" \
  -e LLM_MODEL="openai/kimi-k2-0711-preview" \
  -e WORKSPACE_BASE=/opt/workspace/agent_workdir \
  -e MAX_ITERATIONS=10 \
  -e OBSERVER_API_URL="http://host.docker.internal:18858" \
  -e OBSERVER_SESSION_ID="my-rollout-001" \
  -e OBSERVER_SESSION_LABEL="softmax-operator-test" \
  -v /home/l00619433/rllm_ascend/examples/openhands_sdk/workspace:/opt/workspace \
  --network host \
  --add-host host.docker.internal:host-gateway \
  --entrypoint /opt/workspace/entrypoint.py \
  -e http_proxy="http://p_atlas:proxy%40123@80.253.20.124:8080" \
  -e https_proxy="http://p_atlas:proxy%40123@80.253.20.124:8080" \
  -e no_proxy=127.0.0.1,.huawei.com,localhost,local,.local,.docker.internal \
  openhands-triton-env:v1
FROM python:3.12-slim

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
    /etc/apt/sources.list.d/debian.sources \
 && apt-get update -qq \
 && apt-get install -y -qq curl git ca-certificates xz-utils ripgrep \
 && rm -rf /var/lib/apt/lists/*

ARG NODE_VERSION=22.20.0
RUN mkdir -p /opt/benchflow/node /opt/benchflow/js-agents /opt/benchflow/bin \
 && node_arch="$(dpkg --print-architecture)" \
 && if [ "$node_arch" = amd64 ]; then node_arch=x64; fi \
 && curl -fsSLo /tmp/node.tar.xz \
    "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
 && tar -xJf /tmp/node.tar.xz -C /opt/benchflow/node \
    --strip-components=1 --no-same-owner \
 && rm /tmp/node.tar.xz

ENV PATH="/opt/benchflow/node/bin:${PATH}"

RUN /opt/benchflow/node/bin/npm install -g \
    --prefix /opt/benchflow/js-agents opencode-ai@1.18.11 \
 && chmod -R a+rX /opt/benchflow

RUN pip install --no-cache-dir -q anthropic openai google-genai

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /verifier /tests

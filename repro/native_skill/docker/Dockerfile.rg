FROM eduskillbench-opencode-runtime:1.18.11

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
    /etc/apt/sources.list.d/debian.sources \
 && apt-get update -qq \
 && apt-get install -y -qq ripgrep \
 && rm -rf /var/lib/apt/lists/*

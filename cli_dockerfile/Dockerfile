FROM jamwojt/ffmpeg_env:8

WORKDIR /app

COPY package/e3k /app/e3k
COPY package/tests /app/tests
COPY e3k.sh /usr/local/bin/e3k

RUN chmod +x /usr/local/bin/e3k

ENTRYPOINT ["/bin/bash"]

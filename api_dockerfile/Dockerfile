FROM jamwojt/ffmpeg_env:7

WORKDIR /app

COPY package/e3k /app/
COPY models /app

VOLUME [ "/app/models" ]

EXPOSE 8001

CMD [ "python", "/app/app.py" ]
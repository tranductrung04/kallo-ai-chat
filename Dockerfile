FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user pyproject.toml .
RUN pip install --user --no-cache-dir .
COPY --chown=user . .
RUN sed -i 's/\r$//' scripts/entrypoint.sh

EXPOSE 7860
CMD ["bash", "scripts/entrypoint.sh"]

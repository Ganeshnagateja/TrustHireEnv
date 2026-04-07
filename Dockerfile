FROM python:3.11-slim

LABEL maintainer="TrustHireEnv Contributors"
LABEL description="OpenEnv-compliant multimodal interview integrity benchmark"
LABEL version="1.0.1"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from env.environment import TrustHireEnv; env=TrustHireEnv(difficulty='easy'); env.reset(); print('Docker smoke-test OK')"

CMD python baseline_eval.py --no-llm --episodes 3 && tail -f /dev/null

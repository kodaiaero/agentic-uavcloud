# sandbox/kafka/consumer.py
import sys
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
import json

# /Users/.../agentic-uavcloud ???
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.file_analysis import set_target_dir, validate_data_consistency

consumer = KafkaConsumer(
    "diagnosis-request",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="diag-group",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

try:
    for message in consumer:
        target_dir = message.value["dir"]
        set_target_dir(target_dir)
        results = validate_data_consistency.invoke({})
        results["request_id"] = message.value.get("request_id")
        producer.send("diagnosis-result", results)
        producer.flush()
except KeyboardInterrupt:
    pass
finally:
    producer.close()
    consumer.close()




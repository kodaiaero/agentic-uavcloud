from kafka import KafkaProducer, KafkaConsumer
import json
import uuid

bootstrap = "localhost:9092"
request_topic = "diagnosis-request"
result_topic = "diagnosis-result"

request_id = str(uuid.uuid4())

producer = KafkaProducer(
    bootstrap_servers = bootstrap,
    value_serializer = lambda v: json.dumps(v).encode("utf-8"),
)

producer.send(request_topic, {"type": "diagnosis", "dir": "/tmp/drone_data", "request_id": request_id})
producer.flush()

consumer = KafkaConsumer(
    result_topic,
    bootstrap_servers = bootstrap,
    auto_offset_reset = "latest",
    enable_auto_commit = True,
    group_id = f"{request_id}",
    value_deserializer = lambda b: json.loads(b.decode("utf-8"))
)

for message in consumer:
    payload = message.value
    if payload.get("request_id") == request_id:
        print ("result", payload)
        break

consumer.close()
producer.close()


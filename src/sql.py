import random
from datetime import datetime, timedelta

# 生成批量插入数据
data = []
start_date = datetime(2024, 10, 1)
end_date = datetime(2024, 11, 2)
days = (end_date - start_date).days

for day in range(days):
    current_date = start_date + timedelta(days=day)
    for _ in range(10):  # 每天10条数,
        user_id = random.choice([1, 2])
        device_id = random.randint(1, 3)
        start_time = current_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        usage_time = random.randint(1, 240)  # 使用时间范围在1到240分钟之间
        end_time = start_time + timedelta(minutes=usage_time)
        last_updated = datetime.now()

        # 格式化为 SQL 插入语句
        data.append(f"INSERT INTO device_usage (user_id, device_id, start_time, end_time, last_updated, usage_time) "
                    f"VALUES ({user_id}, {device_id}, '{start_time}', '{end_time}', '{last_updated}', {usage_time});")

# 写入 SQL 文件
with open('insert_device_usage.sql', 'w') as f:
    for line in data:
        f.write(line + '\n')

print("SQL 文件 'insert_device_usage.sql' 已生成，包含插入语句。")





fo = open("prediction_32.txt", "r")
contents = fo.readlines()

throughput_025 = []
throughput_05 = []
throughput_075 = []
throughput_1 = []

latency_025 = []
latency_05 = []
latency_075 = []
latency_1 = []

pertime025 = []
pertime05 = []
pertime075 = []
pertime1 = []

averageBatchT025 = []
averageBatchT05 = []
averageBatchT075 = []
averageBatchT1 = []

batchThrough025 = []
batchThrough05 = []
batchThrough075 = []
batchThrough1 = []

X = []

for ele in contents:
    ele = ele.strip()

    if "When num_img/batch_size=" in ele:
        X.append(int(ele[24:]))


    a ="sr_idx=0.25 average_time="
    b ="sr_idx=0.25 total_time="
    c ="sr_idx=0.25 throughput="
    d ="sr_idx=0.25 average_batch_time="
    e ="sr_idx=0.25 throughput_batch="

    if a in ele:
        pertime025.append(float(ele[len(a):]))
    if b in ele:
        latency_025.append(float(ele[len(b):]))
    if c in ele:
        throughput_025.append(float(ele[len(c):])*1000)
    if d in ele:
        averageBatchT025.append(float(ele[len(d):]))
    if e in ele:
        batchThrough025.append(float(ele[len(e):])*1000)


    a ="sr_idx=0.5 average_time="
    b ="sr_idx=0.5 total_time="
    c ="sr_idx=0.5 throughput="
    d ="sr_idx=0.5 average_batch_time="
    e ="sr_idx=0.5 throughput_batch="

    if a in ele:
        pertime05.append(float(ele[len(a):]))
    if b  in ele:
        latency_05.append(float(ele[len(b):]))
    if c in ele:
        throughput_05.append(float(ele[len(c):])*1000)
    if d in ele:
        averageBatchT05.append(float(ele[len(d):]))
    if e in ele:
        batchThrough05.append(float(ele[len(e):])*1000)



    a ="sr_idx=0.75 average_time="
    b ="sr_idx=0.75 total_time="
    c ="sr_idx=0.75 throughput="
    d ="sr_idx=0.75 average_batch_time="
    e ="sr_idx=0.75 throughput_batch="

    if a in ele:
        pertime075.append(float(ele[len(a):]))
    if b in ele:
        latency_075.append(float(ele[len(b):]))
    if c in ele:
        throughput_075.append(float(ele[len(c):])*1000)
    if d in ele:
        averageBatchT075.append(float(ele[len(d):]))
    if e in ele:
        batchThrough075.append(float(ele[len(e):])*1000)


    a = "sr_idx=1.0 average_time="
    b = "sr_idx=1.0 total_time="
    c = "sr_idx=1.0 throughput="
    d = "sr_idx=1.0 average_batch_time="
    e = "sr_idx=1.0 throughput_batch="
    if a in ele:
        pertime1.append(float(ele[len(a):]))
    if b in ele:
        latency_1.append(float(ele[len(b):]))
    if c in ele:
        throughput_1.append(float(ele[len(c):])*1000)
    if d in ele:
        averageBatchT1.append(float(ele[len(d):]))
    if e in ele:
        batchThrough1.append(float(ele[len(e):])*1000)



print("X->")
print(X)

print("Throughput->")
print("a=",throughput_025)
print("b=",throughput_05)
print("c=",throughput_075)
print("d=",throughput_1)

print("Latency->")
print("a=",latency_025)
print("b=",latency_05)
print("c=",latency_075)
print("d=",latency_1)

print("BatchThroughput->")
print("a=",batchThrough025)
print("b=",batchThrough05)
print("c=",batchThrough075)
print("d=",batchThrough1)


print("AverageTime->")
print("a=",pertime025)
print("b=",pertime05)
print("c=",pertime075)
print("d=",pertime1)

print("AverageBatchTime->")
print("a=",averageBatchT025)
print("b=",averageBatchT05)
print("c=",averageBatchT075)
print("d=",averageBatchT1)

fo.close()





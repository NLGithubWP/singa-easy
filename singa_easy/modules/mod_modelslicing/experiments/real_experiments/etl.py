



fo = open("prediction_res1.txt", "r")
fo2 = open("prediction_res2.txt", "r")
fo3 = open("prediction_res3.txt", "r")
content1 = fo.readlines()
content2 = fo2.readlines()
content3 = fo3.readlines()

contents = content1+content2+content3


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

X = []

for ele in contents:
    ele = ele.strip()

    if "When num_img=" in ele:
        X.append(int(ele[13:]))

    if "sr_idx=0.25 average_time=" in ele:
        pertime025.append(float(ele[25:]))
    if "sr_idx=0.25 total_time=" in ele:
        latency_025.append(float(ele[23:]))
    if "sr_idx=0.25 throughput=" in ele:
        throughput_025.append(float(ele[23:])*1000)

    if "sr_idx=0.5 average_time=" in ele:
        pertime05.append(float(ele[25:]))
    if "sr_idx=0.5 total_time=" in ele:
        latency_05.append(float(ele[22:]))
    if "sr_idx=0.5 throughput=" in ele:
        throughput_05.append(float(ele[22:])*1000)


    if "sr_idx=0.75 average_time=" in ele:
        pertime075.append(float(ele[25:]))
    if "sr_idx=0.75 total_time=" in ele:
        latency_075.append(float(ele[23:]))
    if "sr_idx=0.75 throughput=" in ele:
        throughput_075.append(float(ele[23:])*1000)

    if "sr_idx=1.0 average_time=" in ele:
        pertime1.append(float(ele[24:]))
    if "sr_idx=1.0 total_time=" in ele:
        latency_1.append(float(ele[22:]))
    if "sr_idx=1.0 throughput=" in ele:
        throughput_1.append(float(ele[22:])*1000)



print("X->")
print(X)

print("Throughput->")
print(throughput_025)
print(throughput_05)
print(throughput_075)
print(throughput_1)

print("Latency->")
print(latency_025)
print(latency_05)
print(latency_075)
print(latency_1)

print("AverageTime->")
print(pertime025)
print(pertime05)
print(pertime075)
print(pertime1)


fo.close()
fo2.close()
fo3.close()




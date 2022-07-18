import pandas as pd
#  list = [25481.3828125, 22603.787109375, 19070.509765625, 16942.880859375, 14881.7587890625, 12901.9228515625, 11069.7275390625, 9405.5322265625, 7805.79296875, 6473.70166015625, 5183.93408203125, 4133.5673828125, 3232.099365234375, 2505.6083984375, 1929.3887939453125, 1513.789306640625, 1235.3731689453125, 1102.216552734375, 1005.919921875, 1040.7447509765625, 1126.3963623046875, 1217.9619140625, 1366.2578125, 1485.1063232421875, 1651.5025634765625, 1726.38818359375, 1818.3375244140625, 1802.137939453125, 1777.3583984375, 1819.4239501953125, 1715.1380615234375, 1616.0162353515625, 1547.5853271484375, 1439.03564453125, 1366.326171875, 1307.926025390625, 1190.0887451171875, 1118.5361328125, 1063.1129150390625, 1006.0444946289062, 964.884765625, 926.4238891601562, 930.3399047851562, 
# 944.4434814453125, 926.1965942382812, 956.9046020507812, 967.2523803710938, 963.2305908203125, 985.507080078125, 998.3587646484375, 998.1190795898438, 1012.3153686523438, 1015.3648681640625, 973.7544555664062, 991.86767578125, 965.6809692382812, 964.263427734375, 957.4321899414062, 928.8106689453125, 934.833984375, 919.150146484375, 915.4107666015625, 888.8823852539062, 893.4052734375, 898.1488647460938, 901.1773681640625, 896.8441162109375, 874.6482543945312, 887.2449340820312, 890.9226684570312, 896.5118408203125, 910.9705810546875, 886.0718383789062, 893.7549438476562, 882.0353393554688, 873.3673095703125, 878.1316528320312, 861.8663330078125, 890.4047241210938, 863.0429077148438, 868.6173095703125, 857.8455200195312, 840.8552856445312, 849.4345703125, 843.6207885742188, 876.6175537109375, 850.7070922851562, 858.1239624023438, 857.33642578125, 852.1365966796875, 835.3446655273438, 829.4492797851562, 853.7080078125, 856.5853881835938, 820.0158081054688, 828.5941162109375, 828.6080322265625, 823.8792724609375, 812.4277954101562, 835.89111328125, 789.4873657226562, 809.8533935546875, 809.1475830078125, 804.06298828125, 798.9273071289062, 784.1868896484375, 796.3384399414062, 782.9835205078125, 784.2179565429688, 790.2428588867188, 780.3092651367188, 787.288330078125, 761.9812622070312, 761.752685546875, 747.0628662109375, 757.2916259765625, 754.1047973632812, 734.67578125, 733.4974365234375, 733.43701171875, 713.530029296875, 731.18359375, 730.396728515625, 725.9307250976562, 733.091796875, 702.435546875, 702.7609252929688, 679.2408447265625, 689.3422241210938, 687.47900390625, 671.8921508789062, 670.1315307617188, 664.6829223632812, 649.9910888671875, 653.1718139648438, 638.1510009765625, 623.7339477539062, 615.5408935546875, 612.0889282226562, 602.6063842773438, 597.0422973632812, 611.2764892578125, 580.0274047851562, 558.6898193359375, 568.9153442382812, 547.9576416015625, 565.4114379882812, 531.9234619140625, 534.50390625, 516.2632446289062, 509.2312927246094, 519.0380859375, 506.912841796875, 487.8291931152344, 473.5350341796875, 462.29266357421875, 478.7652893066406, 444.0462341308594, 433.2823486328125, 412.7015075683594, 413.84423828125, 416.38079833984375, 400.9496765136719, 400.11376953125, 363.08355712890625, 378.0289001464844, 358.0090637207031, 344.5118408203125, 345.9931640625, 343.4243469238281, 
# 308.87396240234375, 318.15802001953125, 312.2216491699219, 291.17828369140625, 290.8898620605469, 273.8997497558594, 289.0310974121094, 274.3354187011719, 252.07801818847656,
# 264.9789123535156, 261.3702392578125, 260.9852600097656, 248.7478790283203, 239.35736083984375, 227.19639587402344, 243.35487365722656, 238.3944091796875, 218.1646728515625, 
# 215.97572326660156, 218.6331329345703, 221.9900665283203, 201.6761932373047, 210.71131896972656, 209.2412109375, 203.4219512939453, 205.6576385498047, 193.7238311767578,
# 207.62625122070312, 201.30899047851562, 206.9019775390625, 193.31344604492188, 189.12574768066406, 180.58209228515625, 196.67739868164062, 198.561279296875, 182.3700714111328, 
# 189.20384216308594, 194.0005340576172, 186.48509216308594, 186.20587158203125, 184.01995849609375, 190.33578491210938, 170.85867309570312, 176.74026489257812, 184.58999633789062,
# 192.11422729492188, 186.43321228027344, 180.20578002929688, 185.14944458007812, 189.14837646484375, 177.03707885742188, 192.85903930664062, 185.2493896484375, 191.653564453125,
# 182.65809631347656, 182.2460479736328, 186.11570739746094, 189.42684936523438, 174.7563934326172, 184.19528198242188, 182.75193786621094, 193.9688720703125, 179.5480499267578, 
# 180.87640380859375, 176.7150421142578, 192.41082763671875, 188.954833984375, 186.70608520507812, 187.20071411132812, 189.5594940185547, 187.3491668701172, 177.09251403808594, 
# 199.86619567871094, 186.843505859375, 174.71246337890625, 188.91050720214844, 182.2639617919922, 183.60813903808594, 176.16787719726562, 190.5928192138672, 186.37538146972656, 
# 181.96621704101562, 186.08340454101562, 191.30885314941406, 185.76133728027344, 187.33853149414062, 176.97959899902344, 185.7023162841797, 174.68414306640625,
# 181.37884521484375, 196.89410400390625, 178.09097290039062, 183.1805877685547, 186.80691528320312, 190.41099548339844, 188.01315307617188, 194.5917510986328, 
# 187.2438507080078, 189.2440185546875, 194.59336853027344, 195.35073852539062, 170.5649871826172, 186.18634033203125, 181.69630432128906, 183.12554931640625, 179.8224334716797,
# 187.2786102294922, 201.0256805419922, 195.536865234375, 198.01502990722656, 181.47340393066406, 184.8486328125, 194.55516052246094, 187.48866271972656, 176.01919555664062,
# 187.77810668945312, 175.22142028808594, 185.48553466796875, 185.36065673828125, 179.39723205566406, 192.2006378173828, 192.63494873046875, 178.26356506347656, 
# 172.0248260498047, 192.58798217773438, 183.64598083496094, 184.8181915283203, 182.05157470703125, 191.58453369140625, 180.98719787597656]



df = pd.read_csv("data\FeatureData.csv")
df["Close"] = df["Close"]
print(df)
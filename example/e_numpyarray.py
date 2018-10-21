import numpy as np

a = np.zeros((200,1))
b = [[(448.17151, 232.96149), (448.15845, 232.89507), (448.16541, 232.91119), (448.17902, 232.92981), (448.2067, 232.93216), (456.10965, 244.88335), (452.29535, 239.89775), (443.84293, 220.89056), (439.48724, 218.04492), (436.38043, 215.88055)], [(1128.918, 525.91797), (1128.8831, 525.79114), (1128.9095, 525.9046), (1129.0469, 525.94977), (1129.0072, 525.94452), (1128.9675, 525.82452), (1129.0021, 525.82959), (1129.002, 525.82959), (1129.0365, 525.93988), (1129.0365, 525.93988)], [(351.09158, 137.87982), (351.28967, 137.73663), (351.15759, 137.6844), (351.17218, 137.58394), (351.25449, 137.55775), (351.23682, 137.50961), (351.15891, 137.44492), (351.15698, 137.42934), (351.1568, 137.41663), (351.21677, 137.36067)], [(354.2876, 159.69975), (354.30887, 159.70755), (354.25903, 159.69939), (354.32666, 159.73676), (354.27872, 159.73117), (361.16501, 163.68228), (356.1954, 161.04349), (356.14117, 161.01918), (356.1478, 161.02692), (356.10742, 160.94252)], [(593.88562, 497.91168), (593.8847, 497.91116), (593.88605, 497.91168), (593.87933, 497.90704), (593.87909, 497.90802), (593.74902, 497.87854), (593.81775, 497.65219), (593.74908, 497.88577), (593.81854, 497.90741), (593.88995, 497.92651)], [(318.85632, 143.98657), (318.87738, 143.97723), (318.83633, 143.97752), (318.84375, 143.99275), (318.84238, 143.99231), (318.82205, 143.97708), (318.84152, 143.97632), (318.81552, 143.97794), (318.81354, 143.97841), (318.81555, 143.97803)], [(552.08771, 527.72412), (552.07837, 527.77777), (552.08228, 527.90411), (552.20929, 528.23486), (552.33795, 528.44366), (552.37872, 528.44684), (552.3504, 528.34412), (552.35852, 528.35449), (552.33746, 528.33685), (552.43225, 528.40424)], [(555.33398, 479.05377), (555.29553, 479.01428), (555.34033, 479.03024), (555.14874, 479.20282), (555.06311, 479.15903), (555.02429, 479.13599), (555.08795, 478.92746), (555.03192, 479.13162), (555.19037, 479.14615), (555.18109, 479.16849)], [(340.10187, 149.98135), (340.17743, 149.98096), (340.1011, 149.9837), (340.27621, 150.03812), (340.33459, 150.03761), (340.10336, 149.9267), (340.14801, 149.8988), (340.08221, 149.91064), (340.03534, 149.93332), (340.13855, 149.90042)], [(319.1022, 136.42825), (319.09723, 136.42941), (319.05191, 136.43243), (319.14481, 136.43074), (319.13303, 136.43359), (319.15277, 136.42772), (319.16592, 136.42531), (319.15118, 136.42683), (319.15353, 136.42625), (319.1622, 136.42302)], [(715.78571, 470.82498), (715.79993, 470.79141), (715.78998, 470.82303), (716.01593, 470.84714), (715.74902, 470.83569), (715.74872, 470.83554), (715.74683, 470.83618), (715.74664, 470.83621), (715.74915, 470.83371), (715.99884, 470.83691)], [(378.78458, 179.243), (378.85785, 179.30136), (378.97708, 179.23277), (375.83823, 177.06204), (381.8877, 181.27521), (380.78421, 180.47624), (380.80255, 180.46857), (380.7435, 180.43304), (380.74612, 180.41977), (378.79904, 179.03387)], [(546.8504, 503.2413), (546.93042, 503.25565), (546.91852, 503.28601), (547.07629, 503.59634), (547.06592, 503.36768), (546.80823, 503.59183), (546.80853, 503.59464), (546.88043, 503.58951), (546.95599, 503.63092), (546.97876, 503.40485)], [(605.13025, 467.10217), (605.17383, 466.96393), (605.05994, 467.01492), (604.95715, 467.04764), (604.95941, 467.04791), (604.92285, 467.04385), (604.92316, 467.04498), (604.92059, 467.04379), (604.9248, 467.04437), (604.9256, 467.04446)], [(1195.5511, 647.93536), (1195.5543, 647.93701), (1195.5526, 647.93597), (1195.6147, 647.94257), (1195.6146, 647.94244), (1195.3743, 647.94672), (1195.6211, 647.95074), (1195.621, 647.95074), (1195.621, 647.95074), (1195.6237, 647.95178)], [(531.61401, 483.67773), (531.78912, 483.66156), (531.61938, 483.67542), (531.61011, 483.61807), (531.61243, 483.62268), (531.37848, 483.62225), (531.58301, 483.61844), (531.58435, 483.6257), (531.61786, 483.65051), (531.60693, 483.82205)], [(460.98227, 278.31616), (461.00772, 278.34482), (460.98761, 278.35818), (461.00165, 278.29477), (455.06339, 255.71988), (455.85626, 252.10307), (454.07498, 249.55009), (450.95792, 239.39088), (457.58945, 264.40622), (457.60672, 264.59009)], [(563.0, 258.0), (569.8205, 243.35857)], [(426.0, 181.0), (413.3092, 180.73315)], [(576.0, 259.0), (567.70984, 246.53693)], [(563.0, 266.0), (567.86584, 247.01944)], [(513.0, 236.0), (525.74713, 242.91895)], [(575.0, 271.0), (589.56927, 258.9335)], [(701.0, 241.0), (696.36823, 238.49463)], [(401.0, 193.0), (401.13968, 193.10983)], [(675.0, 227.0), (661.58051, 219.93826)], [(539.0, 227.0), (526.37415, 237.12967)], [(492.0, 237.0), (520.3996, 237.55287)], [(553.0, 258.0), (578.91541, 258.58771)], [(441.0, 188.0), (429.33591, 187.76801)], [(452.0, 202.0), (439.83749, 201.75899)], [(452.0, 190.0), (440.25043, 189.64168)], [(627.0, 281.0), (606.31927, 272.09775)], [(665.0, 583.0), (665.01184, 582.98859)], [(556.0, 269.0), (575.62402, 265.33478)], [(584.0, 467.0), (583.90302, 467.01239)], [(684.0, 245.0), (669.49078, 244.16335)], [(513.0, 264.0), (510.70673, 270.28839)], [(500.0, 239.0), (528.55048, 239.42523)], [(412.0, 181.0), (413.76227, 178.85957)], [(585.0, 530.0), (585.03943, 530.02167)], [(431.0, 195.0), (419.18582, 195.08937)], [(518.0, 253.0), (513.48712, 249.19995)], [(479.0, 487.0), (479.00491, 486.98651)], [(484.0, 249.0), (498.25784, 278.20139)], [(738.0, 496.0), (737.99963, 496.0)]]

# b = np.array(b)
# print(b.shape)
print(b.items() > [462, 409])
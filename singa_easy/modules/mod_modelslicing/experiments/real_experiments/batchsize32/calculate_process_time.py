


a = [3.884737014770508, 1.2697822153568268, 0.9717427074909211, 0.7039628386497497, 0.6471758673588435, 0.5718508496880531, 0.6001009956002236, 0.5526740156114102, 0.5085533133574894, 0.5121074987575411, 0.4892803071439266, 0.46597280278801917, 0.46066358877079827, 0.4892564352353414, 0.4809651936404407, 0.4619848701883765, 0.48078910526302127, 0.46587941408157346, 0.48785404896736145, 0.45820637136697767, 0.45010345437697, 0.45263760671019554, 0.439309657977687, 0.45594978432763705, 0.45075570431848366, 0.45017562022575963, 0.46579545878938267, 0.4548853411277135, 0.45946037884801627]
b = [6.125722885131836, 1.781985604763031, 1.1628497242927551, 1.0379760533571243, 0.8976150035858155, 0.8453156508505344, 0.82048952460289, 0.769951444864273, 0.7634706833532878, 0.7156324610114098, 0.6899977520108223, 0.7065996115406354, 0.7066495844296047, 0.6998853113253911, 0.6880362248048186, 0.6845683257369434, 0.6871079986294111, 0.6679002456367016, 0.6721133316755294, 0.6692213792602221, 0.6441736921242305, 0.6564445018023253, 0.6398195927672916, 0.6448909240961075, 0.6433371489743391, 0.6414940862013744, 0.6509061290536609, 0.6480858149925868, 0.6405341540277004]
c = [9.201130867004395, 2.4165753602981566, 1.910624384880066, 1.5959385931491852, 1.437363624572754, 1.3141708813607693, 1.2506139528751374, 1.208153601984183, 1.1028022634131567, 1.1336086977273225, 1.0961371746659279, 1.0693176592389741, 1.0462506324052812, 1.056499068538348, 1.0541116258129477, 1.0514840993811103, 0.9947957264052497, 1.0301133608818054, 0.9983064379692078, 0.9937255103389422, 0.9661141458579472, 0.9649257156252861, 0.9689170111550225, 0.9857365251671184, 0.9699716582894325, 1.001702062350053, 0.9568423290337835, 0.9343874958356222, 0.9752624222263694]
d = [13.450600624084473, 3.439836573600769, 2.4142834544181824, 2.237034922838211, 1.9104412913322448, 1.6975177496671676, 1.611757776737213, 1.5197684278090795, 1.570067127261843, 1.475460411608219, 1.3826661014556885, 1.343834282954534, 1.3533175992114204, 1.3631203587849934, 1.3859556756913662, 1.3125499178381528, 1.303364560339186, 1.2839888328313827, 1.3412441174983978, 1.265968180100123, 1.2793709870747159, 1.3211884726583958, 1.3131663558218214, 1.2552262754873795, 1.2269811736543974, 1.2327284018809979, 1.2838566600424903, 1.2272440088589986, 1.2859540244936942]


ave_a = 0
ave_b = 0
ave_c = 0
ave_d = 0

for ele in a[4:-4]:
    ave_a += ele

for ele in b[4:-4]:
    ave_b += ele

for ele in c[4:-4]:
    ave_c += ele

for ele in d[4:-4]:
    ave_d += ele

print(ave_a/len(a[4:-4]))
print(ave_b/len(b[4:-4]))
print(ave_c/len(c[4:-4]))
print(ave_d/len(d[4:-4]))

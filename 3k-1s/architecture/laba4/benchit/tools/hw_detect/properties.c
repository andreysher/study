#include "properties.h"
const info_t cpu_data[CPU_DATA_COUNT] = {
{"AMD Athlon.*", 6, 1, "K7", 250, "A(462)"},
{"AMD Athlon.*", 6, 2, "K75", 180, "A(462)"},
{"AMD Duron.*", 6, 3, "Spitfire", 180, "A(462)"},
{"AMD Athlon.*", 6, 4, "Thunderbird", 180, "A(462)"},
{"AMD Athlon.*", 6, 6, "Palomino", 180, "A(462)"},
{"AMD Duron.*", 6, 7, "Morgan", 180, "A(462)"},
{"AMD Athlon.*", 6, 8, "Thoroughbred", 130, "A(462)"},
{"AMD Athlon.*", 6, 10, "Barton/Thorton", 130, "A(462)"},
{"AMD Sempron.*", 6, 6, "Palomino", 180, "A(462)"},
{"AMD Sempron.*", 6, 7, "Morgan", 180, "A(462)"},
{"AMD Sempron.*", 6, 8, "Thoroughbred", 130, "A(462)"},
{"Sempron.*", 6, 10, "Barton/Thorton", 130, "A(462)"},
{"AMD Opteron(tm) Processor 1.*", 15, 39, "Venus", 90, "939"},
{"AMD Opteron(tm) Processor 2.*", 15, 37, "Troy", 90, "940"},
{"AMD Opteron(tm) Processor 8.*", 15, 37, "Athens", 90, "940"},
{"Dual Core AMD Opteron(tm) Processor 1.*", 15, 35, "Denmark", 90, "939"},
{"Dual Core AMD Opteron(tm) Processor 2.*", 15, 33, "Italy", 90, "940"},
{"Dual Core AMD Opteron(tm) Processor 8.*", 15, 33, "Egypt", 90, "940"},
{"Dualcore AMD Opteron(tm) Processor 12.*", 15, 67, "Santa Ana", 90, "AM2"},
{"Dualcore AMD Opteron(tm) Processor [28]2.*", 15, 65, "Santa Rosa", 90, "F(1207)"},
{"AMD Opteron(tm) Processor.*", 15, 5, "Sledgehammer", 130, "940"},
{"AMD Opteron(tm) Processor.*", 15, 21, "unknown AMD K8 (revision SH-D0)", 90, "940"},
{"AMD Athlon(tm) 64 FX.*", 15, 5, "Sledgehammer", 130, "940"},
{"AMD Athlon(tm) 64 FX.*", 15, 7, "Sledgehammer", 130, "939"},
{"AMD Athlon(tm) 64 FX.*", 15, 39, "San Diego", 90, "939"},
{"AMD Athlon(tm) 64 FX.*", 15, 21, "unknown AMD K8 (revision SH-D0)", 90, "940"},
{"AMD Athlon(tm) 64 FX.*", 15, 23, "unknown AMD K8 (revision SH-D0)", 90, "939"},
{"AMD Athlon(tm) 64 FX.*", 15, 35, "Toledo", 90, "939"},
{"AMD Athlon(tm) 64 FX.*", 15, 67, "Windsor", 90, "AM2"},
{"AMD Athlon(tm) 64 FX.*", 15, 65, "Windsor", 90, "F(1207)"},
{"AMD Athlon(tm) 64 FX.*", 15, 193, "Windsor", 90, "F(1207)"},
{"AMD Athlon(tm) 64 Processor.*", 15, 4, "Clawhammer", 130, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 7, "Clawhammer", 130, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 20, "unknown AMD K8 (revision SH-D0)", 90, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 23, "unknown AMD K8 (revision SH-D0)", 90, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 39, "San Diego", 90, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 8, "unknown AMD K8 (revision CH-CG)", 130, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 11, "unknown AMD K8 (revision CH-CG)", 130, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 24, "unknown AMD K8 (revision CH-D0)", 90, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 27, "unknown AMD K8 (revision CH-D0)", 90, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 12, "Newcastle", 130, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 14, "Newcastle", 130, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 15, "Newcastle", 130, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 28, "Winchester", 90, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 31, "Winchester", 90, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 44, "Venice", 90, "754"},
{"AMD Athlon(tm) 64 Processor.*", 15, 47, "Venice", 90, "939"},
{"AMD Athlon(tm) 64 Processor.*", 15, 79, "Orleans", 90, "AM2"},
{"AMD Athlon(tm) 64 Processor.*", 15, 95, "Orleans", 90, "AM2"},
{"AMD Athlon(tm) 64 Processor.*", 15, 76, "Orleans", 90, "S1"},
{"AMD Athlon(tm) 64 Processor.*", 15, 111, "Lima", 65, "AM2"},
{"AMD Athlon(tm) 64 Processor.*", 15, 127, "Lima", 65, "AM2"},
{"AMD Athlon(tm) Processor LE.*", 15, 95, "Orleans", 90, "AM2"},
{"AMD Athlon(tm) 64 X2.*", 15, 43, "Manchester", 90, "939"},
{"AMD Athlon(tm) 64 X2.*", 15, 35, "Toledo", 90, "939"},
{"AMD Athlon(tm) 64 X2.*", 15, 67, "Windsor", 90, "AM2"},
{"AMD Athlon(tm) 64 X2.*", 15, 75, "Windsor", 90, "AM2"},
{"AMD Athlon(tm) 64 X2.*", 15, 104, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) 64 X2.*", 15, 107, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) X2.*", 15, 104, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) X2.*", 15, 107, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) 2.*", 15, 104, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) 2.*", 15, 107, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) Dual.*", 15, 104, "Brisbane", 65, "AM2"},
{"AMD Athlon(tm) Dual.*", 15, 107, "Brisbane", 65, "AM2"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 4, "Clawhammer", 130, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 20, "unknown AMD K8 (revision SH-D0)", 90, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 8, "unknown AMD K8 (revision CH-CG)", 130, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 24, "unknown AMD K8 (revision CH-D0)", 90, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 12, "Odessa", 130, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 14, "Odessa", 130, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 28, "Oakville", 90, "754"},
{"Mobile AMD Athlon(tm) 64 Processor.*", 15, 36, "Newark", 90, "754"},
{"AMD Sempron(tm) Processor.*", 15, 12, "Paris", 130, "754"},
{"AMD Sempron(tm) Processor.*", 15, 14, "Paris", 130, "754"},
{"AMD Sempron(tm) Processor.*", 15, 15, "Paris", 130, "939"},
{"AMD Sempron(tm) Processor.*", 15, 28, "Palermo", 90, "754"},
{"AMD Sempron(tm) Processor.*", 15, 31, "Palermo", 90, "939"},
{"AMD Sempron(tm) Processor.*", 15, 44, "Palermo", 90, "754"},
{"AMD Sempron(tm) Processor.*", 15, 47, "Palermo", 90, "939"},
{"AMD Sempron(tm) Processor.*", 15, 79, "Manila", 90, "AM2"},
{"AMD Sempron(tm) Processor.*", 15, 95, "Manila", 90, "AM2"},
{"AMD Sempron(tm) Processor.*", 15, 76, "Richmond/Keene", 90, "S1"},
{"AMD Sempron(tm) Dual.*", 15, 104, "Brisbane", 65, "AM2"},
{"AMD Sempron(tm) Dual.*", 15, 107, "Brisbane", 65, "AM2"},
{"AMD Sempron(tm) Processor.*", 15, 111, "Sparta/Huron", 65, "n/a"},
{"AMD Sempron(tm) Processor.*", 15, 127, "Sparta/Huron", 65, "n/a"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 8, "Dublin", 130, "754"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 24, "unknown AMD K8 (revision CH-D0)", 90, "754"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 12, "Paris", 130, "754"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 14, "Paris", 130, "754"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 28, "Georgetown", 90, "754"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 44, "Albany", 90, "754"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 76, "Richmond/Keene", 90, "S1"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 108, "Sherman", 65, "S1"},
{"Mobile AMD Sempron(tm) Processor.*", 15, 124, "Sherman", 65, "S1"},
{"Mobile AMD Athlon(tm).*", 15, 108, "Sherman", 65, "S1"},
{"Mobile AMD Athlon(tm).*", 15, 124, "Sherman", 65, "S1"},
{"Mobile AMD Athlon(tm) XP_M Processor.*", 15, 4, "Clawhammer", 130, "754"},
{"Mobile AMD Athlon(tm) XP_M Processor.*", 15, 20, "unknown AMD K8 (revision SH-D0)", 90, "754"},
{"Mobile AMD Athlon(tm) XP-M Processor.*", 15, 8, "Dublin", 130, "754"},
{"Mobile AMD Athlon(tm) XP-M Processor.*", 15, 24, "unknown AMD K8 (revision CH-D0)", 90, "754"},
{"Mobile AMD Athlon(tm) XP-M Processor.*", 15, 12, "Paris", 130, "754"},
{"Mobile AMD Athlon(tm) XP-M Processor.*", 15, 14, "Paris", 130, "754"},
{"Mobile AMD Athlon(tm) XP-M Processor.*", 15, 28, "Georgetown", 90, "754"},
{"AMD Turion(tm) 64 Mobile.*", 15, 36, "Lancaster", 90, "754"},
{"AMD Turion(tm) 64 Mobile.*", 15, 76, "Richmond", 90, "S1"},
{"AMD Turion(tm) 64 X2.*", 15, 72, "Trinidad/Taylor", 90, "S1"},
{"AMD Turion(tm) 64 X2.*", 15, 100, "Tyler", 65, "S1"},
{"AMD Athlon(tm) Neo X2.*", 15, 107, "Conesus", 65, "n/a"},
{"AMD Athlon(tm) Neo.*", 15, 111, "Huron", 65, "ASB1"},
{"AMD Athlon(tm) Neo.*", 15, 127, "Huron", 65, "ASB1"},
{"AMD Turion(tm) Neo X2.*", 15, 107, "Conesus", 65, "ASB1"},
{"Quad-Core AMD Opteron(tm) Processor 13.*", 16, 2, "Budapest", 65, "AM2+"},
{"Quad-Core AMD Opteron(tm) Processor 23.*", 16, 2, "Barcelona", 65, "F(1207)"},
{"Quad-Core AMD Opteron(tm) Processor 83.*", 16, 2, "Barcelona", 65, "F(1207)"},
{"Quad-Core AMD Opteron(tm) Processor 13.*", 16, 4, "Suzuka", 45, "AM3"},
{"Quad-Core AMD Opteron(tm) Processor 23.*", 16, 4, "Shanghai", 45, "F(1207)"},
{"Quad-Core AMD Opteron(tm) Processor 83.*", 16, 4, "Shanghai", 45, "F(1207)"},
{"Six-Core AMD Opteron(tm) Processor 14.*", 16, 8, "unknown AMD K10", 45, "n/a"},
{"Six-Core AMD Opteron(tm) Processor 24.*", 16, 8, "Istanbul", 45, "F(1207)"},
{"Six-Core AMD Opteron(tm) Processor 84.*", 16, 8, "Istanbul", 45, "F(1207)"},
{"AMD Opteron(tm) Processor 41.*", 16, 8, "Lisbon", 45, "C32"},
{"AMD Opteron(tm) Processor 61.*", 16, 9, "Magny-Cours", 45, "G34"},
{"Embedded AMD Opteron(tm) Processor 13.*", 16, 2, "Budapest(embedded)", 65, "n/a"},
{"Embedded AMD Opteron(tm) Processor 23.*", 16, 2, "Barcelona(embedded)", 65, "n/a"},
{"Embedded AMD Opteron(tm) Processor 83.*", 16, 2, "Barcelona(embedded)", 65, "n/a"},
{"Embedded AMD Opteron(tm) Processor 13.*", 16, 4, "Suzuka(embedded)", 45, "n/a"},
{"Embedded AMD Opteron(tm) Processor 23.*", 16, 4, "Shanghai(embedded)", 45, "n/a"},
{"Embedded AMD Opteron(tm) Processor 83.*", 16, 4, "Shanghai(embedded)", 45, "n/a"},
{"Embedded AMD Opteron(tm) Processor 24.*", 16, 8, "Istanbul(embedded)", 45, "n/a"},
{"Embedded AMD Opteron(tm) Processor 84.*", 16, 8, "Istanbul(embedded)", 45, "n/a"},
{"AMD Phenom(tm) 9.*", 16, 2, "Agena", 65, "AM2+"},
{"AMD Phenom(tm) 8.*", 16, 2, "Toliman", 65, "AM2+"},
{"AMD Athlon(tm).*", 16, 2, "Kuma", 65, "AM2+"},
{"AMD Phenom(tm) II X6.*", 16, 10, "Thuban", 45, "AM3"},
{"AMD Phenom(tm) II X4.*", 16, 10, "Zosma", 45, "AM3"},
{"AMD Phenom(tm) II X4.*", 16, 4, "Deneb", 45, "AM3"},
{"AMD Phenom(tm) II X3.*", 16, 4, "Heka", 45, "AM3"},
{"AMD Phenom(tm) II X2.*", 16, 4, "Calisto", 45, "AM3"},
{"AMD Phenom(tm) II 10.*", 16, 10, "Thuban", 45, "AM3"},
{"AMD Phenom(tm) II 9.*", 16, 10, "Zosma", 45, "AM3"},
{"AMD Phenom(tm) II [89].*", 16, 4, "Deneb", 45, "AM3"},
{"AMD Phenom(tm) II 7.*", 16, 4, "Heka", 45, "AM3"},
{"AMD Phenom(tm) II 5.*", 16, 4, "Calisto", 45, "AM3"},
{"AMD Athlon(tm) II X4.*", 16, 5, "Propus", 45, "AM3"},
{"AMD Athlon(tm) II X3.*", 16, 5, "Rana", 45, "AM3"},
{"AMD Athlon(tm) II X2.*", 16, 5, "Propus", 45, "AM3"},
{"AMD Athlon(tm) II X2.*", 16, 6, "Regor", 45, "AM3"},
{"AMD Athlon(tm) II [26].*", 16, 5, "Propus", 45, "AM3"},
{"AMD Athlon(tm) II 4.*", 16, 5, "Rana", 45, "AM3"},
{"AMD Athlon(tm) II [12].*", 16, 6, "Regor", 45, "AM3"},
{"AMD Sempron(tm) X2 1.*", 16, 6, "Regor", 45, "AM3"},
{"AMD Sempron(tm) 1.*", 16, 6, "Regor", 45, "AM3"},
{"AMD Sempron(tm) M1.*", 16, 6, "Caspian", 45, "S1"},
{"AMD Turion(tm) II Ultra.*", 16, 6, "Caspian", 45, "S1"},
{"AMD Turion(tm) II Dual.*", 16, 6, "Caspian", 45, "S1"},
{"AMD Athlon(tm) II Dual.*", 16, 6, "Caspian", 45, "S1"},
{"AMD V.*", 16, 6, "Caspian/Geneva", 45, "n/a"},
{"AMD Turion(tm) II [NP].*", 16, 6, "Caspian", 45, "S1"},
{"AMD Athlon(tm) II [NP].*", 16, 6, "Caspian", 45, "S1"},
{"AMD Phenom(tm) II [NPX].*", 16, 6, "Caspian", 45, "S1"},
{"AMD Phenom(tm) II [NPX].*", 16, 5, "Champlain", 45, "S1"},
{"AMD Athlon(tm) II Neo.*", 16, 6, "Geneva", 45, "ASB2"},
{"AMD Turion(tm) II Neo.*", 16, 6, "Geneva", 45, "ASB2"},
{"AMD Engineering Sample.*", 16, 2, "Barcelona (ES) ", 65, "n/a"},
{"AMD Engineering Sample.*", 16, 4, "Shanghai (ES)", 45, "n/a"},
{"AMD Engineering Sample.*", 16, 5, "Propus (ES)", 45, "n/a"},
{"AMD Engineering Sample.*", 16, 6, "Regor (ES)", 45, "n/a"},
{"AMD Engineering Sample.*", 16, 8, "Istanbul/Lisbon (ES)", 45, "n/a"},
{"AMD Engineering Sample.*", 16, 9, "Magny-Cours (ES)", 45, "n/a"},
{"AMD Engineering Sample.*", 16, 10, "Thuban (ES)", 45, "n/a"},
{"AMD Turion(tm) X2.*", 17, 3, "Lion", 65, "S1"},
{"AMD Athlon(tm) X2.*", 17, 3, "Lion", 65, "S1"},
{"AMD Athlon(tm) QI.*", 17, 3, "Sable", 65, "S1"},
{"AMD Sempron(tm) X2.*", 17, 3, "Lion", 65, "S1"},
{"AMD Sempron(tm) SI.*", 17, 3, "Sable", 65, "S1"},
{"Intel(R) Pentium(R) Pro processor.*", 6, 1, "P6", -1, "n/a"},
{"Intel(R) Pentium(R) II processor.*", 6, 3, "Klamath", 350, "n/a"},
{"Intel(R) Pentium(R) II processor.*", 6, 5, "Deschutes", 250, "n/a"},
{"Intel(R) Celeron(R).*", 6, 6, "Covington", 250, "n/a"},
{"Intel(R) Pentium(R) III processor.*", 6, 7, "Katmai", 250, "n/a"},
{"Intel(R) Pentium(R) III Xeon.*", 6, 7, "Tanner", 250, "n/a"},
{"Intel(R) Celeron(R).*", 6, 7, "Mendocino", 250, "n/a"},
{"Intel(R) Pentium(R) III processor.*", 6, 8, "Coppermine", 180, "n/a"},
{"Intel(R) Pentium(R) III Xeon.*", 6, 8, "Cascades", 180, "n/a"},
{"Intel(R) Celeron(R).*", 6, 8, "Coppermine-128", 180, "n/a"},
{"Mobile Intel(R) Pentium(R) III.*", 6, 9, "Banias", 130, "n/a"},
{"Mobile Intel(R) Celeron(R) III.*", 6, 9, "Banias-512", 130, "n/a"},
{"Intel(R) Pentium(R) M.*", 6, 9, "Banias", 130, "n/a"},
{"Intel(R) Pentium(R) III Xeon.*", 6, 10, "Cascades", 180, "n/a"},
{"Intel(R) Pentium(R) III processor.*", 6, 11, "Tualatin", 130, "n/a"},
{"Intel(R) Celeron(R).*", 6, 11, "Tualatin-256", 130, "n/a"},
{"Intel(R) Pentium(R) M.*", 6, 13, "Dothan", 90, "n/a"},
{"Intel(R) Celeron(R).*", 6, 13, "Dothan", 90, "n/a"},
{"Intel(R) Core(TM) Solo.*", 6, 14, "Yonah", 65, "n/a"},
{"Intel(R) Core(TM) Duo.*", 6, 14, "Yonah", 65, "n/a"},
{"Intel(R) Pentium(R) Dual.*", 6, 14, "Yonah", 65, "n/a"},
{"Intel(R) Celeron.*", 6, 14, "Yonah", 65, "n/a"},
{"Genuine Intel(R).*", 6, 14, "Yonah", 65, "n/a"},
{"Mobile Intel(R) Atom(TM) CPU *N2.*", 6, 28, "Diamondville", 45, "n/a"},
{"Intel(R) Atom(TM) CPU *N?[23].*", 6, 28, "Diamondville", 45, "n/a"},
{"Intel(R) Atom(TM) CPU *Z5.*", 6, 28, "Silverthorne", 45, "n/a"},
{"Mobile Intel(R) Atom(TM) CPU *Z5.*", 6, 28, "Silverthorne", 45, "n/a"},
{"Intel(R) Atom(TM) CPU *D[45].*", 6, 28, "Pineview", 45, "n/a"},
{"Mobile Intel(R) Atom(TM) CPU *N4.*", 6, 28, "Pineview", 45, "n/a"},
{"Intel(R) Atom(TM) CPU *N4.*", 6, 28, "Pineview", 45, "n/a"},
{"Intel(R) Pentium(R) 4.*", 15, 0, "Willamette", 180, "n/a"},
{"Intel(R) Celeron.*", 15, 0, "Willamette-128", 180, "n/a"},
{"Intel(R) Xeon.*", 15, 0, "Foster", 180, "n/a"},
{"Intel(R) Pentium(R) 4.*", 15, 1, "Willamette", 180, "n/a"},
{"Intel(R) Celeron.*", 15, 1, "Willamette-128", 180, "n/a"},
{"Intel(R) Xeon.*", 15, 1, "Foster", 180, "n/a"},
{"Intel(R) Pentium(R) 4 CPU.*", 15, 2, "Northwood", 130, "n/a"},
{"Mobile Intel(R) Pentium(R) 4.*", 15, 2, "Northwood", 130, "n/a"},
{"Intel(R) Pentium(R) 4 Extreme.*", 15, 2, "Gallatin", 130, "n/a"},
{"Intel(R) Celeron.*", 15, 2, "Northwood-128", 130, "n/a"},
{"Intel(R) Xeon.*", 15, 2, "Prestonia/Gallatin", 130, "n/a"},
{"Intel(R) Pentium(R) 4 CPU.*", 15, 3, "Prescott", 90, "n/a"},
{"Mobile Intel(R) Pentium(R) 4.*", 15, 3, "Prescott", 90, "n/a"},
{"Intel(R) Pentium(R) 4 Extreme.*", 15, 3, "Prescott-2M", 90, "n/a"},
{"Intel(R) Pentium(R) D.*", 15, 3, "Smithfield", 90, "n/a"},
{"Intel(R) Celeron.*", 15, 3, "Prescott-256", 90, "n/a"},
{"Intel(R) Xeon(R) CPU              .*", 15, 3, "Nocona/Irwindale/Paxville/Cranford/Potomac", 90, "n/a"},
{"Intel(R) Xeon(R) CPU *70.*", 15, 3, "Paxville MP", 90, "n/a"},
{"Intel(R) Pentium(R) 4 CPU.*", 15, 4, "Prescott", 90, "n/a"},
{"Mobile Intel(R) Pentium(R) 4.*", 15, 3, "Prescott", 90, "n/a"},
{"Intel(R) Pentium(R) 4 Extreme.*", 15, 4, "Prescott-2M", 90, "n/a"},
{"Intel(R) Pentium(R) D.*", 15, 4, "Smithfield", 90, "n/a"},
{"Intel(R) Celeron.*", 15, 4, "Prescott-256", 90, "n/a"},
{"Intel(R) Xeon(R) CPU              .*", 15, 4, "Nocona/Irwindale/Paxville/Cranford/Potomac", 90, "n/a"},
{"Intel(R) Xeon(R) CPU *70.*", 15, 4, "Paxville", 90, "n/a"},
{"Intel(R) Pentium(R) 4 CPU.*", 15, 6, "Cedar Mill", 65, "n/a"},
{"Mobile Intel(R) Pentium(R) 4.*", 15, 6, "Cedar Mill", 65, "n/a"},
{"Intel(R) Pentium(R) 4 Extreme.*", 15, 6, "Cedar Mill", 65, "n/a"},
{"Intel(R) Pentium(R) D.*", 15, 6, "Presler", 65, "n/a"},
{"Intel(R) Celeron.*", 15, 6, "Cedar Mill-512", 65, "n/a"},
{"Intel(R) Xeon(R) CPU *50.*", 15, 6, "Dempsey", 65, "n/a"},
{"Intel(R) Xeon(R) CPU *71.*", 15, 6, "Tulsa", 65, "n/a"},
{"Intel(R) Core(TM)2 Solo.*", 6, 15, "Merom", 65, "n/a"},
{"Intel(R) Core(TM)2 CPU *4.*", 6, 15, "Allendale", 65, "775"},
{"Intel(R) Core(TM)2 CPU *6.*", 6, 15, "Conroe", 65, "775"},
{"Intel(R) Core(TM)2 Duo CPU *E4.*", 6, 15, "Allendale", 65, "775"},
{"Intel(R) Core(TM)2 Duo CPU *E6.*", 6, 15, "Conroe", 65, "775"},
{"Intel(R) Core(TM)2 Duo CPU *[LTU][57].*", 6, 15, "Merom", 65, "n/a"},
{"Intel(R) Core(TM)2 CPU *[LTU][57].*", 6, 15, "Merom", 65, "n/a"},
{"Intel(R) Core(TM)2 Quad CPU    Q6.*", 6, 15, "Kentsfield", 65, "775"},
{"Intel(R) Core(TM)2 Extreme CPU X6.*", 6, 15, "Conroe", 65, "775"},
{"Intel(R) Core(TM)2 Extreme CPU X7.*", 6, 15, "Merom", 65, "n/a"},
{"Intel(R) Core(TM)2 Extreme CPU Q6.*", 6, 15, "Kentsfield", 65, "775"},
{"Intel(R) Pentium(R) Dual CPU   E2.*", 6, 15, "Allendale", 65, "775"},
{"Intel(R) Pentium(R) Dual CPU   T2.*", 6, 15, "Merom", 65, "n/a"},
{"Intel(R) Celeron(R) CPU *[24].*", 6, 15, "Conroe-L", 65, "775"},
{"Intel(R) Celeron(R) CPU *E.*", 6, 15, "Allendale", 65, "775"},
{"Intel(R) Celeron(R) CPU *[T5].*", 6, 15, "Merom", 65, "n/a"},
{"Intel(R) Celeron(R).*", 6, 22, "Merom", 65, "n/a"},
{"Intel(R) Core(TM) Solo.*", 6, 22, "Merom", 65, "n/a"},
{"Intel(R) Core(TM)2 Solo.*", 6, 23, "Penryn-3M", 45, "n/a"},
{"Intel(R) Core(TM)2 Duo CPU *E7.*", 6, 23, "Wolfdale-3M", 45, "775"},
{"Intel(R) Core(TM)2 Duo CPU *E8.*", 6, 23, "Wolfdale", 45, "775"},
{"Intel(R) Core(TM)2 Duo CPU *[LPT]9.*", 6, 23, "Penryn", 45, "n/a"},
{"Intel(R) Core(TM)2 Duo CPU *[PT]8.*", 6, 23, "Penryn-3M", 45, "n/a"},
{"Intel(R) Core(TM)2 Duo CPU *P7.*", 6, 23, "Penryn-3M", 45, "n/a"},
{"Intel(R) Core(TM)2 Duo CPU *U9.*", 6, 23, "Penryn-3M", 45, "n/a"},
{"Intel(R) Core(TM)2 Quad CPU *Q8.*", 6, 23, "Yorkfield-4M", 45, "775"},
{"Intel(R) Core(TM)2 Quad CPU *Q9.*", 6, 23, "Yorkfield", 45, "775"},
{"Intel(R) Core(TM)2 Extreme CPU X9.*", 6, 23, "Penryn", 45, "n/a"},
{"Intel(R) Core(TM)2 Extreme CPU Q9.*", 6, 23, "Yorkfield", 45, "n/a"},
{"Intel(R) Pentium(R) Dual CPU *E[56].*", 6, 23, "Wolfdale-2M", 45, "775"},
{"Intel(R) Celeron(R) CPU *E3.*", 6, 23, "Wolfdale-1M", 45, "775"},
{"Intel(R) Celeron(R) CPU *T3.*", 6, 23, "Penryn-1M", 45, "n/a"},
{"Intel(R) Core(TM) i7 CPU *9.*", 6, 26, "Bloomfield", 45, "1366"},
{"Intel(R) Core(TM) i7 CPU *8.*", 6, 30, "Lynnfield", 45, "1156"},
{"Intel(R) Core(TM) i5 CPU *7.*", 6, 30, "Lynnfield", 45, "1156"},
{"Intel(R) Core(TM) i7 CPU *9.*", 6, 44, "Gulftown", 32, "1366"},
{"Intel(R) Core(TM) i5 CPU *6.*", 6, 37, "Clarkdale", 32, "1156"},
{"Intel(R) Core(TM) i3 CPU *5.*", 6, 37, "Clarkdale", 32, "1156"},
{"Intel(R) Pentium(R).*", 6, 37, "Clarkdale", 32, "1156"},
{"Intel(R) Core(TM) i7 CPU *[7-9].*", 6, 30, "Clarksfield", 45, "n/a"},
{"Intel(R) Core(TM) i7 CPU *6.*", 6, 37, "Arrandale", 32, "n/a"},
{"Intel(R) Core(TM) i5 CPU *[45].*", 6, 37, "Arrandale", 32, "n/a"},
{"Intel(R) Core(TM) i3 CPU *3.*", 6, 37, "Arrandale", 32, "n/a"},
{"Intel(R) Xeon(R) CPU *30.*", 6, 15, "Conroe", 65, "775"},
{"Intel(R) Xeon(R) CPU *[LEX]31.*", 6, 23, "Wolfdale-UP", 45, "775"},
{"Intel(R) Xeon(R) CPU *[LEX]32.*", 6, 15, "Kentsfield", 65, "775"},
{"Intel(R) Xeon(R) CPU *[LEX]33.*", 6, 23, "Yorkfield-UP", 45, "775"},
{"Intel(R) Xeon(R) CPU *[LEXW]34.*", 6, 30, "Lynnfield", 45, "1156"},
{"Intel(R) Xeon(R) CPU *[LEXW]35.*", 6, 26, "Bloomfield", 45, "1366"},
{"Intel(R) Xeon(R) CPU *[EL]C35.*", 6, 30, "Jasper Forest", 45, "n/a"},
{"Intel(R) Celeron(R) CPU *P1.*", 6, 30, "Jasper Forest", 45, "n/a"},
{"Intel(R) Xeon(R) CPU *51.*", 6, 15, "Woodcrest", 65, "771"},
{"Intel(R) Xeon(R) CPU *[LEX]52.*", 6, 23, "Wolfdale-DP", 45, "771"},
{"Intel(R) Xeon(R) CPU *[LEX]53.*", 6, 15, "Clovertown", 65, "771"},
{"Intel(R) Xeon(R) CPU *[LEX]54.*", 6, 23, "Harpertown", 45, "771"},
{"Intel(R) Xeon(R) CPU *[LEXW]55.*", 6, 26, "Gainestown", 45, "1366"},
{"Intel(R) Xeon(R) CPU *[LE]C55.*", 6, 30, "Jasper Forest", 45, "n/a"},
{"Intel(R) Xeon(R) CPU *[LEXW]56.*", 6, 44, "Westmere-EP", 32, "1366"},
{"Intel(R) Xeon(R) CPU *E72.*", 6, 15, "Tigerton-DC", 65, "604"},
{"Intel(R) Xeon(R) CPU *[LEX]73.*", 6, 15, "Tigerton", 65, "604"},
{"Intel(R) Xeon(R) CPU *[LEX]74.*", 6, 23, "Dunnington", 45, "604"},
{"Intel(R) Xeon(R) CPU *[EX]75.*", 6, 46, "Beckton", 45, "1567"},
{"Intel(R) Xeon(R) CPU *E65.*", 6, 46, "Beckton", 45, "1567"},
};
const archshrt_t archshrt_data[ARCH_SHORT_COUNT] = {
{"Celeron", 1, "InCe"},
{"Celeron", 2, "ICe2"},
{"Pentium(R) III", 1, "InP3"},
{"Pentium(R) 4", 1, "InP4"},
{"Pentium(R) M", 1, "InPM"},
{"Pentium(R) Dual", 2, "IPDC"},
{"Pentium(R) D CPU", 2, "InPD"},
{"Core(TM)2", 2, "IC2D"},
{"Core(TM)2", 4, "IC2Q"},
{"Core(TM) Solo", 1, "InCS"},
{"Core(TM) Duo", 2, "InCD"},
{"Core(TM)i7", 6, "ICi7"},
{"Core(TM)i7", 4, "ICi7"},
{"Core(TM)i7", 2, "ICi7"},
{"Core(TM)i5", 4, "ICi5"},
{"Core(TM)i5", 2, "ICi5"},
{"Core(TM)i3", 2, "ICi3"},
{"Xeon", 1, "InXn"},
{"Xeon", 2, "InX2"},
{"Xeon", 4, "InX4"},
{"Xeon", 6, "InX6"},
{"Xeon", 8, "InX8"},
{"Xeon", 16, "IX16"},
{"Opteron(tm)", 1, "AmOp"},
{"Opteron(tm)", 2, "AmO2"},
{"Opteron(tm)", 4, "AmO4"},
{"Opteron(tm)", 6, "AmO6"},
{"Opteron(tm)", 8, "AmO8"},
{"Opteron(tm)", 12, "AO12"},
{"Phenom(tm) II", 3, "APX3"},
{"Phenom(tm) II", 4, "APX4"},
{"Phenom(tm) II", 6, "APX6"},
{"Athlon(tm) 64", 1, "AmK8"},
{"Athlon(tm) 64 ", 2, "AmX2"},
{"Athlon(tm) X2", 2, "AmX2"},
{"Athlon(tm) X3", 3, "AmX3"},
{"Athlon(tm) X4", 4, "AmX4"},
{"Athlon(tm) II", 2, "AmX2"},
{"Athlon(tm) II", 3, "AmX3"},
{"Athlon(tm) II", 4, "AmX4"},
{"Turion(tm)", 1, "AmTu"},
{"Turion(tm)", 2, "ATX2"},
{"Mobile AMD Athlon(tm) XP-M", 1, "AmK8"},
{"Athlon(tm) Processor LE", 1, "AmK8"},
{"Athlon(tm) XP", 1, "AmK7"},
{"Athlon(tm) Processor  ", 1, "AmK7"},
{"Athlon(tm) processor  ", 1, "AmK7"},
{"AMD-K5", 1, "AmK5"},
{"AMD-K6", 1, "AmK6"},
{"AMD-K7", 1, "AmK7"},
{"Duron(tm)", 1, "AmK7"},
{"Sempron(tm)", 2, "ASX2"},
{"Sempron(tm)", 1, "AmSe"},
{"Intel(R)", 1, "In_1"},
{"Intel(R)", 2, "In_2"},
{"Intel(R)", 4, "In_4"},
{"Intel(R)", 6, "In_6"},
{"Intel(R)", 8, "In_8"},
{"Intel(R)", 16, "I_16"},
{"AMD", 1, "Am_1"},
{"AMD", 2, "Am_2"},
{"AMD", 3, "Am_3"},
{"AMD", 4, "Am_4"},
{"AMD", 6, "Am_6"},
{"AMD", 8, "Am_8"},
{"AMD", 12, "A_12"},
{"VIA", 1, "Vi_1"},
{"VIA", 2, "Vi_2"},
{"VIA", 3, "Vi_3"},
{"VIA", 4, "Vi_4"},
};

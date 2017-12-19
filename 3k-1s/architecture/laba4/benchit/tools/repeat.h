/* Based on: Demystifying GPU Microarchitecture through Microbenchmarking
 http://www.stuffedcow.net/research/cudabmk
*/

#define repeat0(S)
#define repeat1(S) S
#define repeat2(S) S S
#define repeat3(S) S S S
#define repeat4(S) S S S S
#define repeat5(S) S S S S S
#define repeat6(S) S S S S S S
#define repeat7(S) S S S S S S S
#define repeat8(S) S S S S S S S S
#define repeat9(S) S S S S S S S S S
#define repeat10(S) S S S S S S S S S S
#define repeat11(S) repeat10(S) S
#define repeat12(S) repeat10(S) S S
#define repeat13(S) repeat10(S) S S S
#define repeat14(S) repeat10(S) S S S S
#define repeat15(S) repeat10(S) S S S S S
#define repeat16(S) repeat10(S) S S S S S S
#define repeat17(S) repeat10(S) S S S S S S S
#define repeat18(S) repeat10(S) S S S S S S S S
#define repeat19(S) repeat10(S) S S S S S S S S S
#define repeat20(S) repeat10(S) repeat10(S)
#define repeat21(S) repeat20(S) S
#define repeat22(S) repeat20(S) S S
#define repeat23(S) repeat20(S) S S S
#define repeat24(S) repeat20(S) S S S S
#define repeat25(S) repeat20(S) S S S S S
#define repeat26(S) repeat20(S) S S S S S S
#define repeat27(S) repeat20(S) S S S S S S S
#define repeat28(S) repeat20(S) S S S S S S S S
#define repeat29(S) repeat20(S) S S S S S S S S S
#define repeat30(S) repeat20(S) repeat10(S)
#define repeat31(S) repeat20(S) repeat10(S) S
#define repeat32(S) repeat20(S) repeat10(S) S S
#define repeat33(S) repeat20(S) repeat10(S) S S S
#define repeat34(S) repeat20(S) repeat10(S) S S S S
#define repeat35(S) repeat20(S) repeat10(S) S S S S S
#define repeat36(S) repeat20(S) repeat10(S) S S S S S S
#define repeat37(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat38(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat39(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat40(S) repeat20(S) repeat20(S)
#define repeat41(S) repeat20(S) repeat20(S) S
#define repeat42(S) repeat20(S) repeat20(S) S S
#define repeat43(S) repeat20(S) repeat20(S) S S S
#define repeat44(S) repeat20(S) repeat20(S) S S S S
#define repeat45(S) repeat20(S) repeat20(S) S S S S S
#define repeat46(S) repeat20(S) repeat20(S) S S S S S S
#define repeat47(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat48(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat49(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat50(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat51(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat52(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat53(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat54(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat55(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat56(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S
#define repeat57(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat58(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat59(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat60(S) repeat20(S) repeat20(S) repeat20(S)
#define repeat61(S) repeat20(S) repeat20(S) repeat20(S) S
#define repeat62(S) repeat20(S) repeat20(S) repeat20(S) S S
#define repeat63(S) repeat20(S) repeat20(S) repeat20(S) S S S
#define repeat64(S) repeat20(S) repeat20(S) repeat20(S) S S S S
#define repeat65(S) repeat20(S) repeat20(S) repeat20(S) S S S S S
#define repeat66(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S
#define repeat67(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat68(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat69(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat70(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat71(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat72(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat73(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat74(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat75(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat76(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S
#define repeat77(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat78(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat79(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat80(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S)
#define repeat81(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S
#define repeat82(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S
#define repeat83(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S
#define repeat84(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S
#define repeat85(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S
#define repeat86(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S
#define repeat87(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat88(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat89(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat90(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat91(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat92(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat93(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat94(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat95(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat96(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S
#define repeat97(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat98(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat99(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat100(S) repeat50(S) repeat50(S)
#define repeat101(S) repeat100(S) S
#define repeat102(S) repeat100(S) S S
#define repeat103(S) repeat100(S) S S S
#define repeat104(S) repeat100(S) S S S S
#define repeat105(S) repeat100(S) S S S S S
#define repeat106(S) repeat100(S) S S S S S S
#define repeat107(S) repeat100(S) S S S S S S S
#define repeat108(S) repeat100(S) S S S S S S S S
#define repeat109(S) repeat100(S) S S S S S S S S S
#define repeat110(S) repeat100(S) repeat10(S)
#define repeat111(S) repeat100(S) repeat10(S) S
#define repeat112(S) repeat100(S) repeat10(S) S S
#define repeat113(S) repeat100(S) repeat10(S) S S S
#define repeat114(S) repeat100(S) repeat10(S) S S S S
#define repeat115(S) repeat100(S) repeat10(S) S S S S S
#define repeat116(S) repeat100(S) repeat10(S) S S S S S S
#define repeat117(S) repeat100(S) repeat10(S) S S S S S S S
#define repeat118(S) repeat100(S) repeat10(S) S S S S S S S S
#define repeat119(S) repeat100(S) repeat10(S) S S S S S S S S S
#define repeat120(S) repeat100(S) repeat20(S)
#define repeat121(S) repeat100(S) repeat20(S) S
#define repeat122(S) repeat100(S) repeat20(S) S S
#define repeat123(S) repeat100(S) repeat20(S) S S S
#define repeat124(S) repeat100(S) repeat20(S) S S S S
#define repeat125(S) repeat100(S) repeat20(S) S S S S S
#define repeat126(S) repeat100(S) repeat20(S) S S S S S S
#define repeat127(S) repeat100(S) repeat20(S) S S S S S S S
#define repeat128(S) repeat100(S) repeat20(S) S S S S S S S S
#define repeat129(S) repeat100(S) repeat20(S) S S S S S S S S S
#define repeat130(S) repeat100(S) repeat20(S) repeat10(S)
#define repeat131(S) repeat100(S) repeat20(S) repeat10(S) S
#define repeat132(S) repeat100(S) repeat20(S) repeat10(S) S S
#define repeat133(S) repeat100(S) repeat20(S) repeat10(S) S S S
#define repeat134(S) repeat100(S) repeat20(S) repeat10(S) S S S S
#define repeat135(S) repeat100(S) repeat20(S) repeat10(S) S S S S S
#define repeat136(S) repeat100(S) repeat20(S) repeat10(S) S S S S S S
#define repeat137(S) repeat100(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat138(S) repeat100(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat139(S) repeat100(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat140(S) repeat100(S) repeat20(S) repeat20(S)
#define repeat141(S) repeat100(S) repeat20(S) repeat20(S) S
#define repeat142(S) repeat100(S) repeat20(S) repeat20(S) S S
#define repeat143(S) repeat100(S) repeat20(S) repeat20(S) S S S
#define repeat144(S) repeat100(S) repeat20(S) repeat20(S) S S S S
#define repeat145(S) repeat100(S) repeat20(S) repeat20(S) S S S S S
#define repeat146(S) repeat100(S) repeat20(S) repeat20(S) S S S S S S
#define repeat147(S) repeat100(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat148(S) repeat100(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat149(S) repeat100(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat150(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat151(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat152(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat153(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat154(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat155(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat156(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S
#define repeat157(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat158(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat159(S) repeat100(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat160(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S)
#define repeat161(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S
#define repeat162(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S
#define repeat163(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S
#define repeat164(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S S
#define repeat165(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S S S
#define repeat166(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S
#define repeat167(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat168(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat169(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat170(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat171(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat172(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat173(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat174(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat175(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat176(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S
#define repeat177(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat178(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat179(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat180(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S)
#define repeat181(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S
#define repeat182(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S
#define repeat183(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S
#define repeat184(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S
#define repeat185(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S
#define repeat186(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S
#define repeat187(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat188(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat189(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat190(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat191(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat192(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat193(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat194(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat195(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat196(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S
#define repeat197(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat198(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat199(S) repeat100(S) repeat20(S) repeat20(S) repeat20(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat200(S) repeat100(S) repeat100(S)
#define repeat201(S) repeat200(S) S
#define repeat202(S) repeat200(S) S S
#define repeat203(S) repeat200(S) S S S
#define repeat204(S) repeat200(S) S S S S
#define repeat205(S) repeat200(S) S S S S S
#define repeat206(S) repeat200(S) S S S S S S
#define repeat207(S) repeat200(S) S S S S S S S
#define repeat208(S) repeat200(S) S S S S S S S S
#define repeat209(S) repeat200(S) S S S S S S S S S
#define repeat210(S) repeat200(S) repeat10(S)
#define repeat211(S) repeat200(S) repeat10(S) S
#define repeat212(S) repeat200(S) repeat10(S) S S
#define repeat213(S) repeat200(S) repeat10(S) S S S
#define repeat214(S) repeat200(S) repeat10(S) S S S S
#define repeat215(S) repeat200(S) repeat10(S) S S S S S
#define repeat216(S) repeat200(S) repeat10(S) S S S S S S
#define repeat217(S) repeat200(S) repeat10(S) S S S S S S S
#define repeat218(S) repeat200(S) repeat10(S) S S S S S S S S
#define repeat219(S) repeat200(S) repeat10(S) S S S S S S S S S
#define repeat220(S) repeat200(S) repeat20(S)
#define repeat221(S) repeat200(S) repeat20(S) S
#define repeat222(S) repeat200(S) repeat20(S) S S
#define repeat223(S) repeat200(S) repeat20(S) S S S
#define repeat224(S) repeat200(S) repeat20(S) S S S S
#define repeat225(S) repeat200(S) repeat20(S) S S S S S
#define repeat226(S) repeat200(S) repeat20(S) S S S S S S
#define repeat227(S) repeat200(S) repeat20(S) S S S S S S S
#define repeat228(S) repeat200(S) repeat20(S) S S S S S S S S
#define repeat229(S) repeat200(S) repeat20(S) S S S S S S S S S
#define repeat230(S) repeat200(S) repeat20(S) repeat10(S)
#define repeat231(S) repeat200(S) repeat20(S) repeat10(S) S
#define repeat232(S) repeat200(S) repeat20(S) repeat10(S) S S
#define repeat233(S) repeat200(S) repeat20(S) repeat10(S) S S S
#define repeat234(S) repeat200(S) repeat20(S) repeat10(S) S S S S
#define repeat235(S) repeat200(S) repeat20(S) repeat10(S) S S S S S
#define repeat236(S) repeat200(S) repeat20(S) repeat10(S) S S S S S S
#define repeat237(S) repeat200(S) repeat20(S) repeat10(S) S S S S S S S
#define repeat238(S) repeat200(S) repeat20(S) repeat10(S) S S S S S S S S
#define repeat239(S) repeat200(S) repeat20(S) repeat10(S) S S S S S S S S S
#define repeat240(S) repeat200(S) repeat20(S) repeat20(S)
#define repeat241(S) repeat200(S) repeat20(S) repeat20(S) S
#define repeat242(S) repeat200(S) repeat20(S) repeat20(S) S S
#define repeat243(S) repeat200(S) repeat20(S) repeat20(S) S S S
#define repeat244(S) repeat200(S) repeat20(S) repeat20(S) S S S S
#define repeat245(S) repeat200(S) repeat20(S) repeat20(S) S S S S S
#define repeat246(S) repeat200(S) repeat20(S) repeat20(S) S S S S S S
#define repeat247(S) repeat200(S) repeat20(S) repeat20(S) S S S S S S S
#define repeat248(S) repeat200(S) repeat20(S) repeat20(S) S S S S S S S S
#define repeat249(S) repeat200(S) repeat20(S) repeat20(S) S S S S S S S S S
#define repeat250(S) repeat200(S) repeat20(S) repeat20(S) repeat10(S)
#define repeat251(S) repeat200(S) repeat20(S) repeat20(S) repeat10(S) S
#define repeat252(S) repeat200(S) repeat20(S) repeat20(S) repeat10(S) S S
#define repeat253(S) repeat200(S) repeat20(S) repeat20(S) repeat10(S) S S S
#define repeat254(S) repeat200(S) repeat20(S) repeat20(S) repeat10(S) S S S S
#define repeat255(S) repeat200(S) repeat20(S) repeat20(S) repeat10(S) S S S S S
#define repeat256(S) repeat128(S) repeat128(S)
#define repeat512(S) repeat256(S) repeat256(S)
#define repeat768(S) repeat512(S) repeat256(S)
#define repeat1024(S) repeat512(S) repeat512(S)
#define repeat1280(S) repeat1024(S) repeat256(S)
#define repeat2048(S) repeat1024(S) repeat1024(S)

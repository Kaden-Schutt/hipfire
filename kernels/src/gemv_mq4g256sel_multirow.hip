// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4-G256-SEL multirow GEMV — qt=43, 64-profile selector family.
// AoS per-group, 132 B/group = 128 B nibble indices + 2 B fp16 scale + 1 B selector + 1 B pad.
// Reconstruction: w = cb[sel][nibble] * scale, cb is 64x16 `SEL_TABLE` (4 KiB constant).
// 4.125 bpw, 4-byte aligned. Selector is per-GROUP (wave-uniform on 32-thread workgroup),
// so the 16 live levels are held in SGPRs via scalar loads — no LDS for codebook.
// Hoist scales to LDS (R*gpr floats) to amortize narrow half loads.
// Selector byte read once per group, not per weight.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __constant__ float SEL_TABLE[1024] = {
    -1.00000000f, -0.63569781f, -0.48256599f, -0.36915170f, -0.27405864f, -0.18973722f, -0.11235291f, -0.03720061f, 0.03739225f, 0.11309603f, 0.19056368f, 0.27459270f, 0.36915774f, 0.48197658f, 0.63408985f, 1.00000000f, // 0
    -1.00000000f, -0.65379187f, -0.50042644f, -0.38059831f, -0.27903120f, -0.18749237f, -0.10168444f, -0.02034936f, 0.05915787f, 0.13856007f, 0.22072594f, 0.30957789f, 0.41289488f, 0.53803198f, 0.71751187f, 1.00000000f, // 1
    -1.00000000f, -0.70653864f, -0.53235974f, -0.41234162f, -0.31403867f, -0.22632638f, -0.14426093f, -0.06413934f, 0.01622733f, 0.09866407f, 0.18424860f, 0.27742854f, 0.38176155f, 0.50977299f, 0.69728468f, 1.00000000f, // 2
    -1.00000000f, -0.72033111f, -0.53817722f, -0.40855769f, -0.29779644f, -0.20056037f, -0.11052891f, -0.02520977f, 0.05608176f, 0.13620761f, 0.21749551f, 0.30397372f, 0.39879048f, 0.50775141f, 0.64961119f, 1.00000000f, // 3
    -1.00000000f, -0.67740298f, -0.52007657f, -0.40363848f, -0.30639530f, -0.21849881f, -0.13419736f, -0.04897614f, 0.03769624f, 0.12630226f, 0.21965873f, 0.31898824f, 0.42520432f, 0.54614099f, 0.69291956f, 1.00000000f, // 4
    -1.00000000f, -0.77373715f, -0.57736838f, -0.43984751f, -0.32891319f, -0.23074631f, -0.14143270f, -0.05745222f, 0.02418185f, 0.10674654f, 0.19526514f, 0.29068003f, 0.39620345f, 0.52306996f, 0.68671683f, 1.00000000f, // 5
    -1.00000000f, -0.66424748f, -0.52868595f, -0.41685218f, -0.31700524f, -0.22280741f, -0.13765253f, -0.05578286f, 0.02523351f, 0.10854822f, 0.20122238f, 0.30284759f, 0.42017389f, 0.56599320f, 0.76188240f, 1.00000000f, // 6
    -1.00000000f, -0.73471836f, -0.54927990f, -0.40976382f, -0.29502229f, -0.19251692f, -0.09954473f, -0.01026990f, 0.07509253f, 0.16024568f, 0.24646567f, 0.33854782f, 0.44200928f, 0.56493025f, 0.74077414f, 1.00000000f, // 7
    -1.00000000f, -0.70907678f, -0.52601665f, -0.39812116f, -0.29495974f, -0.20423092f, -0.11773817f, -0.03304846f, 0.05062836f, 0.13607805f, 0.22594374f, 0.32459706f, 0.43912301f, 0.58557091f, 0.77842244f, 1.00000000f, // 8
    -1.00000000f, -0.71592486f, -0.55879509f, -0.44073362f, -0.33841900f, -0.24412537f, -0.15414503f, -0.06608558f, 0.02216721f, 0.11097657f, 0.20240707f, 0.30214498f, 0.41541700f, 0.54679349f, 0.72519062f, 1.00000000f, // 9
    -1.00000000f, -0.73721136f, -0.57960768f, -0.44606266f, -0.32756078f, -0.22251730f, -0.12785241f, -0.03737620f, 0.05151675f, 0.13861292f, 0.22766248f, 0.31997605f, 0.42054467f, 0.53449168f, 0.68063841f, 1.00000000f, // 10
    -1.00000000f, -0.74738560f, -0.56707508f, -0.43780104f, -0.32513563f, -0.23080969f, -0.14328337f, -0.05866152f, 0.02635582f, 0.11455007f, 0.21004317f, 0.31249411f, 0.43204711f, 0.57600743f, 0.78483840f, 1.00000000f, // 11
    -1.00000000f, -0.71953082f, -0.55783721f, -0.42653267f, -0.31294723f, -0.21231288f, -0.11787176f, -0.02574706f, 0.06511577f, 0.15669810f, 0.25021466f, 0.35015779f, 0.46011910f, 0.59036072f, 0.74164357f, 1.00000000f, // 12
    -1.00000000f, -0.68562477f, -0.54930709f, -0.43818793f, -0.33714749f, -0.24386300f, -0.15010627f, -0.05815858f, 0.03552260f, 0.13156721f, 0.23086858f, 0.33816932f, 0.45577822f, 0.58731907f, 0.75653057f, 1.00000000f, // 13
    -1.00000000f, -0.80576143f, -0.58807409f, -0.44346073f, -0.32434790f, -0.21768217f, -0.12254642f, -0.03184573f, 0.05607400f, 0.14257014f, 0.22993773f, 0.32379325f, 0.42859632f, 0.55562443f, 0.72861218f, 1.00000000f, // 14
    -1.00000000f, -0.76556284f, -0.59710911f, -0.46601024f, -0.35800809f, -0.25991658f, -0.16552361f, -0.07307706f, 0.02081066f, 0.11665622f, 0.21784766f, 0.32149360f, 0.43236883f, 0.55317714f, 0.70203907f, 1.00000000f, // 15
    -1.00000000f, -0.78746881f, -0.59180373f, -0.44311523f, -0.32419400f, -0.22133909f, -0.12946450f, -0.04319529f, 0.04264076f, 0.13315221f, 0.22982336f, 0.33644543f, 0.45527523f, 0.59243946f, 0.80256084f, 1.00000000f, // 16
    -1.00000000f, -0.77550330f, -0.61634802f, -0.47416223f, -0.35463017f, -0.24729821f, -0.14787722f, -0.05636850f, 0.03014853f, 0.11786239f, 0.21157668f, 0.31380153f, 0.43388660f, 0.58047176f, 0.74604026f, 1.00000000f, // 17
    -1.00000000f, -0.72489507f, -0.57889099f, -0.44867084f, -0.33426338f, -0.22574079f, -0.12045141f, -0.02006500f, 0.07413036f, 0.16735782f, 0.26109091f, 0.36037007f, 0.46834508f, 0.59318296f, 0.76494493f, 1.00000000f, // 18
    -1.00000000f, -0.77119925f, -0.57220168f, -0.43339863f, -0.32647014f, -0.22768410f, -0.13513187f, -0.04437755f, 0.04629424f, 0.14139860f, 0.23949210f, 0.34723554f, 0.47315465f, 0.61677723f, 0.77141620f, 1.00000000f, // 19
    -1.00000000f, -0.74453527f, -0.58676692f, -0.46703614f, -0.35652865f, -0.25222891f, -0.15236268f, -0.05665369f, 0.03804546f, 0.13280937f, 0.23173097f, 0.33778303f, 0.46008717f, 0.59922995f, 0.81874212f, 1.00000000f, // 20
    -1.00000000f, -0.78948480f, -0.61537955f, -0.47335888f, -0.34575302f, -0.23627060f, -0.13488804f, -0.03652505f, 0.06027031f, 0.15622999f, 0.25131011f, 0.34866731f, 0.45503461f, 0.57969240f, 0.74099512f, 1.00000000f, // 21
    -1.00000000f, -0.71500007f, -0.56455032f, -0.43877912f, -0.32678610f, -0.22657392f, -0.13191718f, -0.03813573f, 0.05597504f, 0.15227090f, 0.25194018f, 0.36164990f, 0.48801446f, 0.63539210f, 0.82175601f, 1.00000000f, // 22
    -1.00000000f, -0.81751309f, -0.63233843f, -0.49233350f, -0.37345770f, -0.26770343f, -0.16486248f, -0.06650184f, 0.03254544f, 0.13298785f, 0.23504010f, 0.34014465f, 0.45272434f, 0.57992045f, 0.73252905f, 1.00000000f, // 23
    -1.00000000f, -0.79439608f, -0.61609122f, -0.48088997f, -0.36510952f, -0.26239559f, -0.16689381f, -0.07610886f, 0.01562042f, 0.11364500f, 0.21913933f, 0.33472469f, 0.46327974f, 0.61830198f, 0.79841088f, 1.00000000f, // 24
    -1.00000000f, -0.75046240f, -0.58536422f, -0.46330820f, -0.35384781f, -0.25301183f, -0.15126182f, -0.05154604f, 0.04933016f, 0.15472901f, 0.26210273f, 0.37298900f, 0.48561236f, 0.61178642f, 0.76476197f, 1.00000000f, // 25
    -1.00000000f, -0.76192158f, -0.60025124f, -0.45954095f, -0.33448725f, -0.21963937f, -0.11410181f, -0.01371546f, 0.08271919f, 0.17964704f, 0.27719375f, 0.38220341f, 0.49614797f, 0.63054053f, 0.79203693f, 1.00000000f, // 26
    -1.00000000f, -0.84501497f, -0.63797259f, -0.48458345f, -0.35767356f, -0.24323129f, -0.14008833f, -0.03786372f, 0.06327029f, 0.16105992f, 0.25902228f, 0.35956286f, 0.46766868f, 0.60258488f, 0.76979315f, 1.00000000f, // 27
    -1.00000000f, -0.78316190f, -0.62000940f, -0.47768968f, -0.36004226f, -0.25796102f, -0.15934650f, -0.06196803f, 0.03613816f, 0.13874653f, 0.25152589f, 0.37523848f, 0.50048551f, 0.63360805f, 0.80691952f, 1.00000000f, // 28
    -1.00000000f, -0.76310339f, -0.62276108f, -0.49681179f, -0.37894813f, -0.26512692f, -0.15631174f, -0.05092374f, 0.05295794f, 0.15143505f, 0.25321593f, 0.36107581f, 0.48693743f, 0.62457295f, 0.78634980f, 1.00000000f, // 29
    -1.00000000f, -0.80099487f, -0.60414326f, -0.46185550f, -0.35064328f, -0.24353487f, -0.13450230f, -0.02782270f, 0.07461184f, 0.17217789f, 0.27254165f, 0.37809367f, 0.49697198f, 0.64142484f, 0.82641824f, 1.00000000f, // 30
    -1.00000000f, -0.82420730f, -0.64961009f, -0.50102152f, -0.37403693f, -0.25870116f, -0.14435833f, -0.03199492f, 0.07471548f, 0.17708844f, 0.27746538f, 0.37834573f, 0.48423311f, 0.60061137f, 0.76400759f, 1.00000000f, // 31
    -1.00000000f, -0.79576380f, -0.63804872f, -0.50393190f, -0.38942105f, -0.28112823f, -0.17437420f, -0.06781373f, 0.04346901f, 0.15523216f, 0.26513797f, 0.37506349f, 0.49356824f, 0.61312832f, 0.76283890f, 1.00000000f, // 32
    -1.00000000f, -0.82263905f, -0.63586612f, -0.49176357f, -0.35625299f, -0.23993873f, -0.13967991f, -0.04494282f, 0.05485792f, 0.15764881f, 0.26407574f, 0.37709385f, 0.50302588f, 0.64734071f, 0.81468861f, 1.00000000f, // 33
    -1.00000000f, -0.82767985f, -0.66187336f, -0.50993884f, -0.37329675f, -0.24765356f, -0.12891914f, -0.02500421f, 0.07533147f, 0.17375775f, 0.28299802f, 0.39625382f, 0.51616610f, 0.64329790f, 0.78290773f, 1.00000000f, // 34
    -1.00000000f, -0.81310723f, -0.64826405f, -0.51482542f, -0.39920297f, -0.28812374f, -0.18484809f, -0.08342387f, 0.01967861f, 0.12431197f, 0.23440644f, 0.35165145f, 0.48061471f, 0.62674857f, 0.80774456f, 1.00000000f, // 35
    -1.00000000f, -0.75503138f, -0.60448886f, -0.47722038f, -0.35926139f, -0.24576734f, -0.13853736f, -0.03359181f, 0.06944581f, 0.17332983f, 0.28246128f, 0.39832282f, 0.52250730f, 0.66965728f, 0.84097539f, 1.00000000f, // 36
    -1.00000000f, -0.84985151f, -0.67525625f, -0.52158010f, -0.40355090f, -0.28002941f, -0.16172801f, -0.04991435f, 0.06398865f, 0.17174335f, 0.27874925f, 0.38646982f, 0.50393143f, 0.64527990f, 0.80648807f, 1.00000000f, // 37
    -1.00000000f, -0.84650589f, -0.67098705f, -0.51426309f, -0.38413324f, -0.27073562f, -0.16611243f, -0.06714721f, 0.03425250f, 0.13984367f, 0.24981875f, 0.36961583f, 0.50798627f, 0.65953697f, 0.82342818f, 1.00000000f, // 38
    -1.00000000f, -0.79764909f, -0.65140619f, -0.51485901f, -0.39186559f, -0.27927016f, -0.17680213f, -0.07088610f, 0.03582169f, 0.15302197f, 0.27437205f, 0.39871656f, 0.52794711f, 0.67927339f, 0.85132547f, 1.00000000f, // 39
    -1.00000000f, -0.84450188f, -0.63931802f, -0.51119894f, -0.39495328f, -0.28026289f, -0.15701015f, -0.04043562f, 0.06180646f, 0.16081102f, 0.26769711f, 0.38902568f, 0.52271061f, 0.66632862f, 0.85440014f, 1.00000000f, // 40
    -1.00000000f, -0.82004152f, -0.67102828f, -0.53053714f, -0.41109561f, -0.30073577f, -0.18688971f, -0.06954741f, 0.05428923f, 0.16926133f, 0.28110896f, 0.38674133f, 0.49518344f, 0.62523872f, 0.78801111f, 1.00000000f, // 41
    -1.00000000f, -0.80814448f, -0.66133816f, -0.52827027f, -0.39156896f, -0.26851807f, -0.15066556f, -0.03934959f, 0.06806459f, 0.17546717f, 0.29173066f, 0.41650909f, 0.55197894f, 0.68730804f, 0.84176481f, 1.00000000f, // 42
    -1.00000000f, -0.79332868f, -0.67262676f, -0.53206175f, -0.41124250f, -0.29777509f, -0.17576754f, -0.06825041f, 0.04263351f, 0.15742644f, 0.28141872f, 0.41770200f, 0.54133569f, 0.66657376f, 0.81463065f, 1.00000000f, // 43
    -1.00000000f, -0.82053223f, -0.64733437f, -0.49970309f, -0.38094582f, -0.26491895f, -0.15074093f, -0.02950415f, 0.08598985f, 0.19581937f, 0.30523912f, 0.42216738f, 0.54549893f, 0.67852182f, 0.84013027f, 1.00000000f, // 44
    -1.00000000f, -0.79846107f, -0.64700562f, -0.52346083f, -0.40844552f, -0.29684723f, -0.18354803f, -0.06301580f, 0.05815069f, 0.18198139f, 0.30237775f, 0.41304110f, 0.53372934f, 0.67316374f, 0.82270796f, 1.00000000f, // 45
    -1.00000000f, -0.85344015f, -0.68151129f, -0.54737555f, -0.42003320f, -0.29927847f, -0.19315883f, -0.08432524f, 0.02457948f, 0.14078099f, 0.25599379f, 0.37279250f, 0.48962082f, 0.61598923f, 0.78163738f, 1.00000000f, // 46
    -1.00000000f, -0.83681090f, -0.66376378f, -0.51574986f, -0.38480354f, -0.27141499f, -0.16987676f, -0.06733688f, 0.05052183f, 0.17583160f, 0.30778297f, 0.43921026f, 0.57275575f, 0.69366897f, 0.86017611f, 1.00000000f, // 47
    -1.00000000f, -0.86202541f, -0.68576806f, -0.54490771f, -0.42048122f, -0.30415031f, -0.18231557f, -0.05691032f, 0.05663306f, 0.17809076f, 0.29266173f, 0.42721275f, 0.54358830f, 0.68286586f, 0.84283343f, 1.00000000f, // 48
    -1.00000000f, -0.84626945f, -0.70313352f, -0.56124620f, -0.41241291f, -0.28892954f, -0.16282668f, -0.04826052f, 0.06267037f, 0.18508947f, 0.30006826f, 0.40869612f, 0.53389244f, 0.66074788f, 0.82469454f, 1.00000000f, // 49
    -1.00000000f, -0.83404924f, -0.68625129f, -0.56210980f, -0.43886865f, -0.31106648f, -0.18924070f, -0.07190178f, 0.04185254f, 0.15744448f, 0.27730788f, 0.41241907f, 0.56434948f, 0.70343726f, 0.84508864f, 1.00000000f, // 50
    -1.00000000f, -0.83750262f, -0.69599622f, -0.56284902f, -0.43422421f, -0.30842019f, -0.18440264f, -0.06140996f, 0.06129329f, 0.18438287f, 0.30839564f, 0.43414137f, 0.56278955f, 0.69601953f, 0.83763541f, 1.00000000f, // 51
    -1.00000000f, -0.83993296f, -0.69910765f, -0.56594493f, -0.43692775f, -0.31049628f, -0.18570242f, -0.06184802f, 0.06174108f, 0.18568430f, 0.31047378f, 0.43685181f, 0.56589042f, 0.69912902f, 0.84005468f, 1.00000000f, // 52
    -1.00000000f, -0.84236329f, -0.70221907f, -0.56904085f, -0.43963128f, -0.31257238f, -0.18700220f, -0.06228608f, 0.06218886f, 0.18698573f, 0.31255192f, 0.43956225f, 0.56899129f, 0.70223850f, 0.84247395f, 1.00000000f, // 53
    -1.00000000f, -0.84479363f, -0.70533050f, -0.57213676f, -0.44233482f, -0.31464848f, -0.18830198f, -0.06272414f, 0.06263664f, 0.18828716f, 0.31463006f, 0.44227269f, 0.57209216f, 0.70534798f, 0.84489322f, 1.00000000f, // 54
    -1.00000000f, -0.84722397f, -0.70844193f, -0.57523268f, -0.44503836f, -0.31672457f, -0.18960176f, -0.06316220f, 0.06308442f, 0.18958858f, 0.31670821f, 0.44498314f, 0.57519303f, 0.70845747f, 0.84731250f, 1.00000000f, // 55
    -1.00000000f, -0.84965431f, -0.71155335f, -0.57832859f, -0.44774190f, -0.31880067f, -0.19090154f, -0.06360026f, 0.06353220f, 0.19089001f, 0.31878635f, 0.44769358f, 0.57829390f, 0.71156695f, 0.84973177f, 1.00000000f, // 56
    -1.00000000f, -0.85208464f, -0.71466478f, -0.58142451f, -0.45044544f, -0.32087676f, -0.19220132f, -0.06403831f, 0.06397998f, 0.19219144f, 0.32086449f, 0.45040402f, 0.58139478f, 0.71467643f, 0.85215104f, 1.00000000f, // 57
    -1.00000000f, -0.85451498f, -0.71777620f, -0.58452042f, -0.45314898f, -0.32295286f, -0.19350110f, -0.06447637f, 0.06442776f, 0.19349286f, 0.32294263f, 0.45311446f, 0.58449565f, 0.71778592f, 0.85457031f, 1.00000000f, // 58
    -1.00000000f, -0.85694532f, -0.72088763f, -0.58761634f, -0.45585251f, -0.32502895f, -0.19480088f, -0.06491443f, 0.06487554f, 0.19479429f, 0.32502077f, 0.45582490f, 0.58759652f, 0.72089540f, 0.85698958f, 1.00000000f, // 59
    -1.00000000f, -0.85937565f, -0.72399906f, -0.59071225f, -0.45855605f, -0.32710505f, -0.19610066f, -0.06535249f, 0.06532332f, 0.19609572f, 0.32709891f, 0.45853534f, 0.59069739f, 0.72400488f, 0.85940885f, 1.00000000f, // 60
    -1.00000000f, -0.86180599f, -0.72711048f, -0.59380817f, -0.46125959f, -0.32918114f, -0.19740044f, -0.06579055f, 0.06577110f, 0.19739715f, 0.32917705f, 0.46124578f, 0.59379826f, 0.72711437f, 0.86182812f, 1.00000000f, // 61
    -1.00000000f, -0.86423633f, -0.73022191f, -0.59690408f, -0.46396313f, -0.33125724f, -0.19870022f, -0.06622861f, 0.06621889f, 0.19869857f, 0.33125519f, 0.46395623f, 0.59689913f, 0.73022385f, 0.86424740f, 1.00000000f, // 62
    -1.00000000f, -0.86666667f, -0.73333333f, -0.60000000f, -0.46666667f, -0.33333333f, -0.20000000f, -0.06666667f, 0.06666667f, 0.20000000f, 0.33333333f, 0.46666667f, 0.60000000f, 0.73333333f, 0.86666667f, 1.00000000f, // 63
};

// Templated body shared by R=4 and R=8. R=2 has specialized unrolled body below.
template<int R>
__device__ __forceinline__ void gemv_mq4g256sel_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K)
{
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;
    const int gpr = K / 256;
    const int row_bytes = gpr * 132;
    const int boff = tid * 4;

    // LDS: only scales (R*gpr floats). Codebook is wave-uniform scalar -> SGPRs.
    extern __shared__ float lds[];
    float* sc = lds;
    const int total_sc = R * gpr;
    for (int i = tid; i < total_sc; i += 32) {
        int r = i / gpr;
        int g = i % gpr;
        int src_row = row_base + r;
        if (src_row >= M) src_row = row_base;
        const char* gptr = A + (long long)src_row * row_bytes + g * 132;
        __half h = *reinterpret_cast<const __half*>(gptr + 128);
        sc[i] = __half2float(h);
    }
    __syncthreads();

    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        int row = row_base + r;
        int safe = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe * row_bytes;
    }

    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);

        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp0 = row_ptrs[r] + g * 132;
            const char* gp1 = gp0 + 132;
            const char* gp2 = gp1 + 132;
            const char* gp3 = gp2 + 132;

            float sc0 = sc[r * gpr + g + 0];
            float sc1 = sc[r * gpr + g + 1];
            float sc2 = sc[r * gpr + g + 2];
            float sc3 = sc[r * gpr + g + 3];

            // Selector once per group (wave-uniform). Mask to 6 bits; 64..255 invalid.
            unsigned int sel0 = (unsigned int)(*(const unsigned char*)(gp0 + 130));
            unsigned int sel1 = (unsigned int)(*(const unsigned char*)(gp1 + 130));
            unsigned int sel2 = (unsigned int)(*(const unsigned char*)(gp2 + 130));
            unsigned int sel3 = (unsigned int)(*(const unsigned char*)(gp3 + 130));
            // Scalar loads: 16 f32 per group into SGPRs. The index sel*16 is uniform,
            // so the compiler emits s_load_dwordx4 / s_load_b64.
            const float* cb0 = SEL_TABLE + sel0 * 16;
            const float* cb1 = SEL_TABLE + sel1 * 16;
            const float* cb2 = SEL_TABLE + sel2 * 16;
            const float* cb3 = SEL_TABLE + sel3 * 16;
            float c00 = cb0[0], c01 = cb0[1], c02 = cb0[2], c03 = cb0[3], c04 = cb0[4], c05 = cb0[5], c06 = cb0[6], c07 = cb0[7];
            float c08 = cb0[8], c09 = cb0[9], c10 = cb0[10], c11 = cb0[11], c12 = cb0[12], c13 = cb0[13], c14 = cb0[14], c15 = cb0[15];
            float d00 = cb1[0], d01 = cb1[1], d02 = cb1[2], d03 = cb1[3], d04 = cb1[4], d05 = cb1[5], d06 = cb1[6], d07 = cb1[7];
            float d08 = cb1[8], d09 = cb1[9], d10 = cb1[10], d11 = cb1[11], d12 = cb1[12], d13 = cb1[13], d14 = cb1[14], d15 = cb1[15];
            float e00 = cb2[0], e01 = cb2[1], e02 = cb2[2], e03 = cb2[3], e04 = cb2[4], e05 = cb2[5], e06 = cb2[6], e07 = cb2[7];
            float e08 = cb2[8], e09 = cb2[9], e10 = cb2[10], e11 = cb2[11], e12 = cb2[12], e13 = cb2[13], e14 = cb2[14], e15 = cb2[15];
            float f00 = cb3[0], f01 = cb3[1], f02 = cb3[2], f03 = cb3[3], f04 = cb3[4], f05 = cb3[5], f06 = cb3[6], f07 = cb3[7];
            float f08 = cb3[8], f09 = cb3[9], f10 = cb3[10], f11 = cb3[11], f12 = cb3[12], f13 = cb3[13], f14 = cb3[14], f15 = cb3[15];
            // Map nibble to SGPR level via branchless? Use small LUT via if-else chain that compiler can fold to SGPR select?
            // Instead of LDS indirection, use explicit compare chain that stays in VGPR/SGPR but is still branchless via ternary.
            // However 16-way if chain would be divergent; we can use a small SGPR-indexed array as above plus VGPR-indexed load?
            // The above scalar loads put the 16 levels in SGPRs; we still need to select per nibble (VGPR index).
            // We keep the LUT as SGPR values and index via VGPR using a simple switch that the compiler will lower to v_interp or s_select?
            // For now use a helper: we have the 16 values in registers, we can use a 16-entry SGPR array and index with VGPR via
            // a manual unrolled compare that is still per-lane but only 16 compares vs 16 LDS loads — measured to be cheaper than divergence?
            // Alternative: keep a tiny LUT in SGPR and use `__builtin`? We'll use a straightforward if-chain; the selector being uniform means the 16 levels are uniform, but nibble is per-lane divergent, so any selection will be divergent.
            // The GL kernel's LDS approach was branchless via `cbt[q]`; for SEL we can instead use a VGPR-indexed SGPR array via `__builtin_amdgcn_s_load`? Simpler to just use the same LDS idea for the per-group 16 levels: stage the 16 levels for this group into VGPRs and then do a small LUT via ternary — but we already have them in SGPRs, we can just do a 16-way ternary that is similar to the old divergence but now the table is per-group (16) not per-tensor (256). The divergence cost is same as before (16-way) but without LDS.
            // We will keep the explicit 16-way lookup via ternary for clarity; the compiler will emit v_cmpx etc but only for 16 entries, not 256.
            #define LUT_SGPR(val, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) \
                ((val)==0 ? (a0) : (val)==1 ? (a1) : (val)==2 ? (a2) : (val)==3 ? (a3) : (val)==4 ? (a4) : (val)==5 ? (a5) : (val)==6 ? (a6) : (val)==7 ? (a7) : (val)==8 ? (a8) : (val)==9 ? (a9) : (val)==10 ? (a10) : (val)==11 ? (a11) : (val)==12 ? (a12) : (val)==13 ? (a13) : (val)==14 ? (a14) : (a15))

            unsigned int pk0 = *(const unsigned int*)(gp0 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + boff);

            unsigned int q00 =  pk0        & 0xFu;
            unsigned int q01 = (pk0 >> 4)  & 0xFu;
            unsigned int q02 = (pk0 >> 8)  & 0xFu;
            unsigned int q03 = (pk0 >> 12) & 0xFu;
            unsigned int q04 = (pk0 >> 16) & 0xFu;
            unsigned int q05 = (pk0 >> 20) & 0xFu;
            unsigned int q06 = (pk0 >> 24) & 0xFu;
            unsigned int q07 = (pk0 >> 28) & 0xFu;

            unsigned int q10 =  pk1        & 0xFu;
            unsigned int q11 = (pk1 >> 4)  & 0xFu;
            unsigned int q12 = (pk1 >> 8)  & 0xFu;
            unsigned int q13 = (pk1 >> 12) & 0xFu;
            unsigned int q14 = (pk1 >> 16) & 0xFu;
            unsigned int q15 = (pk1 >> 20) & 0xFu;
            unsigned int q16 = (pk1 >> 24) & 0xFu;
            unsigned int q17 = (pk1 >> 28) & 0xFu;

            unsigned int q20 =  pk2        & 0xFu;
            unsigned int q21 = (pk2 >> 4)  & 0xFu;
            unsigned int q22 = (pk2 >> 8)  & 0xFu;
            unsigned int q23 = (pk2 >> 12) & 0xFu;
            unsigned int q24 = (pk2 >> 16) & 0xFu;
            unsigned int q25 = (pk2 >> 20) & 0xFu;
            unsigned int q26 = (pk2 >> 24) & 0xFu;
            unsigned int q27 = (pk2 >> 28) & 0xFu;

            unsigned int q30 =  pk3        & 0xFu;
            unsigned int q31 = (pk3 >> 4)  & 0xFu;
            unsigned int q32 = (pk3 >> 8)  & 0xFu;
            unsigned int q33 = (pk3 >> 12) & 0xFu;
            unsigned int q34 = (pk3 >> 16) & 0xFu;
            unsigned int q35 = (pk3 >> 20) & 0xFu;
            unsigned int q36 = (pk3 >> 24) & 0xFu;
            unsigned int q37 = (pk3 >> 28) & 0xFu;

            float v00 = LUT_SGPR(q00, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v01 = LUT_SGPR(q01, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v02 = LUT_SGPR(q02, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v03 = LUT_SGPR(q03, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v04 = LUT_SGPR(q04, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v05 = LUT_SGPR(q05, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v06 = LUT_SGPR(q06, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v07 = LUT_SGPR(q07, c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);

            float w10 = LUT_SGPR(q10, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w11 = LUT_SGPR(q11, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w12 = LUT_SGPR(q12, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w13 = LUT_SGPR(q13, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w14 = LUT_SGPR(q14, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w15 = LUT_SGPR(q15, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w16 = LUT_SGPR(q16, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w17 = LUT_SGPR(q17, d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);

            float x20 = LUT_SGPR(q20, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x21 = LUT_SGPR(q21, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x22 = LUT_SGPR(q22, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x23 = LUT_SGPR(q23, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x24 = LUT_SGPR(q24, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x25 = LUT_SGPR(q25, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x26 = LUT_SGPR(q26, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x27 = LUT_SGPR(q27, e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);

            float y30 = LUT_SGPR(q30, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y31 = LUT_SGPR(q31, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y32 = LUT_SGPR(q32, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y33 = LUT_SGPR(q33, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y34 = LUT_SGPR(q34, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y35 = LUT_SGPR(q35, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y36 = LUT_SGPR(q36, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y37 = LUT_SGPR(q37, f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            #undef LUT_SGPR

            acc[r][0] += sc0 * (v00*xa0 + v01*xa1 + v02*xa2 + v03*xa3 + v04*xa4 + v05*xa5 + v06*xa6 + v07*xa7);
            acc[r][1] += sc1 * (w10*xb0 + w11*xb1 + w12*xb2 + w13*xb3 + w14*xb4 + w15*xb5 + w16*xb6 + w17*xb7);
            acc[r][2] += sc2 * (x20*xc0 + x21*xc1 + x22*xc2 + x23*xc3 + x24*xc4 + x25*xc5 + x26*xc6 + x27*xc7);
            acc[r][3] += sc3 * (y30*xd0 + y31*xd1 + y32*xd2 + y33*xd3 + y34*xd4 + y35*xd5 + y36*xd6 + y37*xd7);
        }
    }

    #undef LOAD_X8_INTO

    #define TAIL_LOAD_X(b) \
        const float4 _tv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 _tv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float _x0 = _tv0.x, _x1 = _tv0.y, _x2 = _tv0.z, _x3 = _tv0.w; \
        const float _x4 = _tv1.x, _x5 = _tv1.y, _x6 = _tv1.z, _x7 = _tv1.w

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp = row_ptrs[r] + g * 132;
            float scc = sc[r * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu;
            unsigned int qq1 = (pk >> 4)  & 0xFu;
            unsigned int qq2 = (pk >> 8)  & 0xFu;
            unsigned int qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu;
            unsigned int qq5 = (pk >> 20) & 0xFu;
            unsigned int qq6 = (pk >> 24) & 0xFu;
            unsigned int qq7 = (pk >> 28) & 0xFu;
            acc[r][0] += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp = row_ptrs[r] + g * 132;
            float scc = sc[r * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu;
            unsigned int qq1 = (pk >> 4)  & 0xFu;
            unsigned int qq2 = (pk >> 8)  & 0xFu;
            unsigned int qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu;
            unsigned int qq5 = (pk >> 20) & 0xFu;
            unsigned int qq6 = (pk >> 24) & 0xFu;
            unsigned int qq7 = (pk >> 28) & 0xFu;
            acc[r][1] += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const char* gp = row_ptrs[r] + g * 132;
            float scc = sc[r * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu;
            unsigned int qq1 = (pk >> 4)  & 0xFu;
            unsigned int qq2 = (pk >> 8)  & 0xFu;
            unsigned int qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu;
            unsigned int qq5 = (pk >> 20) & 0xFu;
            unsigned int qq6 = (pk >> 24) & 0xFu;
            unsigned int qq7 = (pk >> 28) & 0xFu;
            acc[r][2] += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
    }
    #undef TAIL_LOAD_X

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] = s;
    }
}

// Specialized R=2 body
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4g256sel_multirow_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K)
{
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;
    const int gpr = K / 256;
    const int row_bytes = gpr * 132;
    const int boff = tid * 4;

    extern __shared__ float lds[];
    float* sc = lds;
    for (int i = tid; i < 2 * gpr; i += 32) {
        int r = i / gpr;
        int g = i % gpr;
        int src_row = (r == 0) ? row0 : (have_row1 ? row1 : row0);
        const char* gptr = A + (long long)src_row * row_bytes + g * 132;
        __half h = *reinterpret_cast<const __half*>(gptr + 128);
        sc[i] = __half2float(h);
    }
    __syncthreads();

    const char* row_ptr0 = A + (long long)row0 * row_bytes;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_bytes;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);

        // Row 0
        {
            const char* gp0 = row_ptr0 + g * 132;
            const char* gp1 = gp0 + 132;
            const char* gp2 = gp1 + 132;
            const char* gp3 = gp2 + 132;
            float sc0 = sc[0 * gpr + g + 0];
            float sc1 = sc[0 * gpr + g + 1];
            float sc2 = sc[0 * gpr + g + 2];
            float sc3 = sc[0 * gpr + g + 3];
            unsigned int sel0 = (unsigned int)(*(const unsigned char*)(gp0 + 130));
            unsigned int sel1 = (unsigned int)(*(const unsigned char*)(gp1 + 130));
            unsigned int sel2 = (unsigned int)(*(const unsigned char*)(gp2 + 130));
            unsigned int sel3 = (unsigned int)(*(const unsigned char*)(gp3 + 130));
            const float* cb0 = SEL_TABLE + sel0 * 16;
            const float* cb1 = SEL_TABLE + sel1 * 16;
            const float* cb2 = SEL_TABLE + sel2 * 16;
            const float* cb3 = SEL_TABLE + sel3 * 16;
            float c00 = cb0[0], c01 = cb0[1], c02 = cb0[2], c03 = cb0[3], c04 = cb0[4], c05 = cb0[5], c06 = cb0[6], c07 = cb0[7];
            float c08 = cb0[8], c09 = cb0[9], c10 = cb0[10], c11 = cb0[11], c12 = cb0[12], c13 = cb0[13], c14 = cb0[14], c15 = cb0[15];
            float d00 = cb1[0], d01 = cb1[1], d02 = cb1[2], d03 = cb1[3], d04 = cb1[4], d05 = cb1[5], d06 = cb1[6], d07 = cb1[7];
            float d08 = cb1[8], d09 = cb1[9], d10 = cb1[10], d11 = cb1[11], d12 = cb1[12], d13 = cb1[13], d14 = cb1[14], d15 = cb1[15];
            float e00 = cb2[0], e01 = cb2[1], e02 = cb2[2], e03 = cb2[3], e04 = cb2[4], e05 = cb2[5], e06 = cb2[6], e07 = cb2[7];
            float e08 = cb2[8], e09 = cb2[9], e10 = cb2[10], e11 = cb2[11], e12 = cb2[12], e13 = cb2[13], e14 = cb2[14], e15 = cb2[15];
            float f00 = cb3[0], f01 = cb3[1], f02 = cb3[2], f03 = cb3[3], f04 = cb3[4], f05 = cb3[5], f06 = cb3[6], f07 = cb3[7];
            float f08 = cb3[8], f09 = cb3[9], f10 = cb3[10], f11 = cb3[11], f12 = cb3[12], f13 = cb3[13], f14 = cb3[14], f15 = cb3[15];
            #define LUT_SGPR(v,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) ((v)==0?a0:(v)==1?a1:(v)==2?a2:(v)==3?a3:(v)==4?a4:(v)==5?a5:(v)==6?a6:(v)==7?a7:(v)==8?a8:(v)==9?a9:(v)==10?a10:(v)==11?a11:(v)==12?a12:(v)==13?a13:(v)==14?a14:a15)
            unsigned int pk0 = *(const unsigned int*)(gp0 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + boff);
            unsigned int q00 =  pk0        & 0xFu, q01 = (pk0 >> 4)  & 0xFu, q02 = (pk0 >> 8)  & 0xFu, q03 = (pk0 >> 12) & 0xFu;
            unsigned int q04 = (pk0 >> 16) & 0xFu, q05 = (pk0 >> 20) & 0xFu, q06 = (pk0 >> 24) & 0xFu, q07 = (pk0 >> 28) & 0xFu;
            unsigned int q10 =  pk1        & 0xFu, q11 = (pk1 >> 4)  & 0xFu, q12 = (pk1 >> 8)  & 0xFu, q13 = (pk1 >> 12) & 0xFu;
            unsigned int q14 = (pk1 >> 16) & 0xFu, q15 = (pk1 >> 20) & 0xFu, q16 = (pk1 >> 24) & 0xFu, q17 = (pk1 >> 28) & 0xFu;
            unsigned int q20 =  pk2        & 0xFu, q21 = (pk2 >> 4)  & 0xFu, q22 = (pk2 >> 8)  & 0xFu, q23 = (pk2 >> 12) & 0xFu;
            unsigned int q24 = (pk2 >> 16) & 0xFu, q25 = (pk2 >> 20) & 0xFu, q26 = (pk2 >> 24) & 0xFu, q27 = (pk2 >> 28) & 0xFu;
            unsigned int q30 =  pk3        & 0xFu, q31 = (pk3 >> 4)  & 0xFu, q32 = (pk3 >> 8)  & 0xFu, q33 = (pk3 >> 12) & 0xFu;
            unsigned int q34 = (pk3 >> 16) & 0xFu, q35 = (pk3 >> 20) & 0xFu, q36 = (pk3 >> 24) & 0xFu, q37 = (pk3 >> 28) & 0xFu;
            float v00 = LUT_SGPR(q00,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v01 = LUT_SGPR(q01,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v02 = LUT_SGPR(q02,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v03 = LUT_SGPR(q03,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v04 = LUT_SGPR(q04,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v05 = LUT_SGPR(q05,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v06 = LUT_SGPR(q06,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v07 = LUT_SGPR(q07,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float w10 = LUT_SGPR(q10,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w11 = LUT_SGPR(q11,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w12 = LUT_SGPR(q12,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w13 = LUT_SGPR(q13,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w14 = LUT_SGPR(q14,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w15 = LUT_SGPR(q15,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w16 = LUT_SGPR(q16,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w17 = LUT_SGPR(q17,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float x20 = LUT_SGPR(q20,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x21 = LUT_SGPR(q21,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x22 = LUT_SGPR(q22,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x23 = LUT_SGPR(q23,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x24 = LUT_SGPR(q24,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x25 = LUT_SGPR(q25,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x26 = LUT_SGPR(q26,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x27 = LUT_SGPR(q27,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float y30 = LUT_SGPR(q30,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y31 = LUT_SGPR(q31,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y32 = LUT_SGPR(q32,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y33 = LUT_SGPR(q33,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y34 = LUT_SGPR(q34,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y35 = LUT_SGPR(q35,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y36 = LUT_SGPR(q36,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y37 = LUT_SGPR(q37,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            #undef LUT_SGPR
            a0 += sc0 * (v00*xa0 + v01*xa1 + v02*xa2 + v03*xa3 + v04*xa4 + v05*xa5 + v06*xa6 + v07*xa7);
            a1 += sc1 * (w10*xb0 + w11*xb1 + w12*xb2 + w13*xb3 + w14*xb4 + w15*xb5 + w16*xb6 + w17*xb7);
            a2 += sc2 * (x20*xc0 + x21*xc1 + x22*xc2 + x23*xc3 + x24*xc4 + x25*xc5 + x26*xc6 + x27*xc7);
            a3 += sc3 * (y30*xd0 + y31*xd1 + y32*xd2 + y33*xd3 + y34*xd4 + y35*xd5 + y36*xd6 + y37*xd7);
        }
        if (have_row1) {
            const char* gp0 = row_ptr1 + g * 132;
            const char* gp1 = gp0 + 132;
            const char* gp2 = gp1 + 132;
            const char* gp3 = gp2 + 132;
            float sc0 = sc[1 * gpr + g + 0];
            float sc1 = sc[1 * gpr + g + 1];
            float sc2 = sc[1 * gpr + g + 2];
            float sc3 = sc[1 * gpr + g + 3];
            unsigned int sel0 = (unsigned int)(*(const unsigned char*)(gp0 + 130));
            unsigned int sel1 = (unsigned int)(*(const unsigned char*)(gp1 + 130));
            unsigned int sel2 = (unsigned int)(*(const unsigned char*)(gp2 + 130));
            unsigned int sel3 = (unsigned int)(*(const unsigned char*)(gp3 + 130));
            const float* cb0 = SEL_TABLE + sel0 * 16;
            const float* cb1 = SEL_TABLE + sel1 * 16;
            const float* cb2 = SEL_TABLE + sel2 * 16;
            const float* cb3 = SEL_TABLE + sel3 * 16;
            float c00 = cb0[0], c01 = cb0[1], c02 = cb0[2], c03 = cb0[3], c04 = cb0[4], c05 = cb0[5], c06 = cb0[6], c07 = cb0[7];
            float c08 = cb0[8], c09 = cb0[9], c10 = cb0[10], c11 = cb0[11], c12 = cb0[12], c13 = cb0[13], c14 = cb0[14], c15 = cb0[15];
            float d00 = cb1[0], d01 = cb1[1], d02 = cb1[2], d03 = cb1[3], d04 = cb1[4], d05 = cb1[5], d06 = cb1[6], d07 = cb1[7];
            float d08 = cb1[8], d09 = cb1[9], d10 = cb1[10], d11 = cb1[11], d12 = cb1[12], d13 = cb1[13], d14 = cb1[14], d15 = cb1[15];
            float e00 = cb2[0], e01 = cb2[1], e02 = cb2[2], e03 = cb2[3], e04 = cb2[4], e05 = cb2[5], e06 = cb2[6], e07 = cb2[7];
            float e08 = cb2[8], e09 = cb2[9], e10 = cb2[10], e11 = cb2[11], e12 = cb2[12], e13 = cb2[13], e14 = cb2[14], e15 = cb2[15];
            float f00 = cb3[0], f01 = cb3[1], f02 = cb3[2], f03 = cb3[3], f04 = cb3[4], f05 = cb3[5], f06 = cb3[6], f07 = cb3[7];
            float f08 = cb3[8], f09 = cb3[9], f10 = cb3[10], f11 = cb3[11], f12 = cb3[12], f13 = cb3[13], f14 = cb3[14], f15 = cb3[15];
            #define LUT_SGPR(v,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) ((v)==0?a0:(v)==1?a1:(v)==2?a2:(v)==3?a3:(v)==4?a4:(v)==5?a5:(v)==6?a6:(v)==7?a7:(v)==8?a8:(v)==9?a9:(v)==10?a10:(v)==11?a11:(v)==12?a12:(v)==13?a13:(v)==14?a14:a15)
            unsigned int pk0 = *(const unsigned int*)(gp0 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + boff);
            unsigned int q00 =  pk0        & 0xFu, q01 = (pk0 >> 4)  & 0xFu, q02 = (pk0 >> 8)  & 0xFu, q03 = (pk0 >> 12) & 0xFu;
            unsigned int q04 = (pk0 >> 16) & 0xFu, q05 = (pk0 >> 20) & 0xFu, q06 = (pk0 >> 24) & 0xFu, q07 = (pk0 >> 28) & 0xFu;
            unsigned int q10 =  pk1        & 0xFu, q11 = (pk1 >> 4)  & 0xFu, q12 = (pk1 >> 8)  & 0xFu, q13 = (pk1 >> 12) & 0xFu;
            unsigned int q14 = (pk1 >> 16) & 0xFu, q15 = (pk1 >> 20) & 0xFu, q16 = (pk1 >> 24) & 0xFu, q17 = (pk1 >> 28) & 0xFu;
            unsigned int q20 =  pk2        & 0xFu, q21 = (pk2 >> 4)  & 0xFu, q22 = (pk2 >> 8)  & 0xFu, q23 = (pk2 >> 12) & 0xFu;
            unsigned int q24 = (pk2 >> 16) & 0xFu, q25 = (pk2 >> 20) & 0xFu, q26 = (pk2 >> 24) & 0xFu, q27 = (pk2 >> 28) & 0xFu;
            unsigned int q30 =  pk3        & 0xFu, q31 = (pk3 >> 4)  & 0xFu, q32 = (pk3 >> 8)  & 0xFu, q33 = (pk3 >> 12) & 0xFu;
            unsigned int q34 = (pk3 >> 16) & 0xFu, q35 = (pk3 >> 20) & 0xFu, q36 = (pk3 >> 24) & 0xFu, q37 = (pk3 >> 28) & 0xFu;
            float v00 = LUT_SGPR(q00,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v01 = LUT_SGPR(q01,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v02 = LUT_SGPR(q02,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v03 = LUT_SGPR(q03,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v04 = LUT_SGPR(q04,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v05 = LUT_SGPR(q05,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v06 = LUT_SGPR(q06,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float v07 = LUT_SGPR(q07,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15);
            float w10 = LUT_SGPR(q10,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w11 = LUT_SGPR(q11,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w12 = LUT_SGPR(q12,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w13 = LUT_SGPR(q13,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w14 = LUT_SGPR(q14,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w15 = LUT_SGPR(q15,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w16 = LUT_SGPR(q16,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float w17 = LUT_SGPR(q17,d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15);
            float x20 = LUT_SGPR(q20,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x21 = LUT_SGPR(q21,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x22 = LUT_SGPR(q22,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x23 = LUT_SGPR(q23,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x24 = LUT_SGPR(q24,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x25 = LUT_SGPR(q25,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x26 = LUT_SGPR(q26,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float x27 = LUT_SGPR(q27,e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15);
            float y30 = LUT_SGPR(q30,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y31 = LUT_SGPR(q31,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y32 = LUT_SGPR(q32,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y33 = LUT_SGPR(q33,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y34 = LUT_SGPR(q34,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y35 = LUT_SGPR(q35,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y36 = LUT_SGPR(q36,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            float y37 = LUT_SGPR(q37,f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15);
            #undef LUT_SGPR
            b0 += sc0 * (v00*xa0 + v01*xa1 + v02*xa2 + v03*xa3 + v04*xa4 + v05*xa5 + v06*xa6 + v07*xa7);
            b1 += sc1 * (w10*xb0 + w11*xb1 + w12*xb2 + w13*xb3 + w14*xb4 + w15*xb5 + w16*xb6 + w17*xb7);
            b2 += sc2 * (x20*xc0 + x21*xc1 + x22*xc2 + x23*xc3 + x24*xc4 + x25*xc5 + x26*xc6 + x27*xc7);
            b3 += sc3 * (y30*xd0 + y31*xd1 + y32*xd2 + y33*xd3 + y34*xd4 + y35*xd5 + y36*xd6 + y37*xd7);
        }
    }

    #undef LOAD_X8_INTO

    #define TAIL_LOAD_X(b) \
        const float4 _tv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 _tv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float _x0 = _tv0.x, _x1 = _tv0.y, _x2 = _tv0.z, _x3 = _tv0.w; \
        const float _x4 = _tv1.x, _x5 = _tv1.y, _x6 = _tv1.z, _x7 = _tv1.w

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        {
            const char* gp = row_ptr0 + g * 132;
            float scc = sc[0 * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu, qq1 = (pk >> 4)  & 0xFu, qq2 = (pk >> 8)  & 0xFu, qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu, qq5 = (pk >> 20) & 0xFu, qq6 = (pk >> 24) & 0xFu, qq7 = (pk >> 28) & 0xFu;
            a0 += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
        if (have_row1) {
            const char* gp = row_ptr1 + g * 132;
            float scc = sc[1 * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu, qq1 = (pk >> 4)  & 0xFu, qq2 = (pk >> 8)  & 0xFu, qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu, qq5 = (pk >> 20) & 0xFu, qq6 = (pk >> 24) & 0xFu, qq7 = (pk >> 28) & 0xFu;
            b0 += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        {
            const char* gp = row_ptr0 + g * 132;
            float scc = sc[0 * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu, qq1 = (pk >> 4)  & 0xFu, qq2 = (pk >> 8)  & 0xFu, qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu, qq5 = (pk >> 20) & 0xFu, qq6 = (pk >> 24) & 0xFu, qq7 = (pk >> 28) & 0xFu;
            a1 += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
        if (have_row1) {
            const char* gp = row_ptr1 + g * 132;
            float scc = sc[1 * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu, qq1 = (pk >> 4)  & 0xFu, qq2 = (pk >> 8)  & 0xFu, qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu, qq5 = (pk >> 20) & 0xFu, qq6 = (pk >> 24) & 0xFu, qq7 = (pk >> 28) & 0xFu;
            b1 += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        {
            const char* gp = row_ptr0 + g * 132;
            float scc = sc[0 * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu, qq1 = (pk >> 4)  & 0xFu, qq2 = (pk >> 8)  & 0xFu, qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu, qq5 = (pk >> 20) & 0xFu, qq6 = (pk >> 24) & 0xFu, qq7 = (pk >> 28) & 0xFu;
            a2 += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
        if (have_row1) {
            const char* gp = row_ptr1 + g * 132;
            float scc = sc[1 * gpr + g];
            unsigned int sel = (unsigned int)(*(const unsigned char*)(gp + 130));
            const float* cb = SEL_TABLE + sel * 16;
            float c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3], c4 = cb[4], c5 = cb[5], c6 = cb[6], c7 = cb[7];
            float c8 = cb[8], c9 = cb[9], c10 = cb[10], c11 = cb[11], c12 = cb[12], c13 = cb[13], c14 = cb[14], c15 = cb[15];
            #define LUT1(v) ((v)==0?c0:(v)==1?c1:(v)==2?c2:(v)==3?c3:(v)==4?c4:(v)==5?c5:(v)==6?c6:(v)==7?c7:(v)==8?c8:(v)==9?c9:(v)==10?c10:(v)==11?c11:(v)==12?c12:(v)==13?c13:(v)==14?c14:c15)
            unsigned int pk = *(const unsigned int*)(gp + boff);
            unsigned int qq0 =  pk        & 0xFu, qq1 = (pk >> 4)  & 0xFu, qq2 = (pk >> 8)  & 0xFu, qq3 = (pk >> 12) & 0xFu;
            unsigned int qq4 = (pk >> 16) & 0xFu, qq5 = (pk >> 20) & 0xFu, qq6 = (pk >> 24) & 0xFu, qq7 = (pk >> 28) & 0xFu;
            b2 += scc * (LUT1(qq0)*_x0 + LUT1(qq1)*_x1 + LUT1(qq2)*_x2 + LUT1(qq3)*_x3 + LUT1(qq4)*_x4 + LUT1(qq5)*_x5 + LUT1(qq6)*_x6 + LUT1(qq7)*_x7);
            #undef LUT1
        }
    }
    #undef TAIL_LOAD_X

    float sa = (a0 + a1) + (a2 + a3);
    float sb = (b0 + b1) + (b2 + b3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sa += __shfl_down(sa, offset);
        sb += __shfl_down(sb, offset);
    }
    if (tid == 0) {
        y[row0] = sa;
        if (have_row1) y[row1] = sb;
    }
}

extern "C" __global__ void gemv_mq4g256sel_multirow_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K)
{
    gemv_mq4g256sel_multirow_body<4>(A, x, y, M, K);
}

extern "C" __global__ void gemv_mq4g256sel_multirow_r8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K)
{
    gemv_mq4g256sel_multirow_body<8>(A, x, y, M, K);
}

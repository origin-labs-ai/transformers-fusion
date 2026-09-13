# Transcender vs Industrial Baselines — REAL Measured Charts

> Generated from `bench_format_comparison.csv` by `tools/generate_comparison_visuals.cpp`.
> Every number below comes from a measured round-trip of the production codec.
> BPW ironclad: each format stores EXACTLY the BPW in its name.

## Gaussian weights — PSNR vs BPW

<svg xmlns='http://www.w3.org/2000/svg' width='900' height='460' font-family='monospace' font-size='10'>
<rect width='100%' height='100%' fill='#fafafa'/>
<line x1='56' y1='404' x2='844' y2='404' stroke='#ddd'/>
<text x='4' y='407'>8 dB</text>
<text x='44' y='418'>1 bpw</text>
<line x1='56' y1='317' x2='844' y2='317' stroke='#ddd'/>
<text x='4' y='320'>33 dB</text>
<text x='241' y='418'>8.75 bpw</text>
<line x1='56' y1='230' x2='844' y2='230' stroke='#ddd'/>
<text x='4' y='233'>59 dB</text>
<text x='438' y='418'>16.5 bpw</text>
<line x1='56' y1='143' x2='844' y2='143' stroke='#ddd'/>
<text x='4' y='146'>85 dB</text>
<text x='635' y='418'>24.25 bpw</text>
<line x1='56' y1='56' x2='844' y2='56' stroke='#ddd'/>
<text x='4' y='59'>110 dB</text>
<text x='832' y='418'>32 bpw</text>
<line x1='56' y1='404' x2='844' y2='404' stroke='#333'/>
<line x1='56' y1='28' x2='56' y2='404' stroke='#333'/>
<circle cx='844' cy='92' r='3' fill='#06c'/>
<text x='849' y='88' fill='#06c'>Q32</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>QG24</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>QG_24.5</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>Q24.5</text>
<circle cx='653' cy='222' r='3' fill='#c6a700'/>
<text x='658' y='218' fill='#c6a700'>Q_MX_24.5</text>
<circle cx='653' cy='222' r='3' fill='#c6a700'/>
<text x='658' y='218' fill='#c6a700'>QG_MX_24.5</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>QG_24_K_H</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>QG_24_K_M</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>QG_24_K_L</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_L</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_M</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_H</text>
<circle cx='450' cy='83' r='3' fill='#06c'/>
<text x='455' y='79' fill='#06c'>QG16</text>
<circle cx='450' cy='83' r='3' fill='#06c'/>
<text x='455' y='79' fill='#06c'>Q16.5</text>
<circle cx='450' cy='83' r='3' fill='#06c'/>
<text x='455' y='79' fill='#06c'>QG_16.5</text>
<circle cx='450' cy='243' r='3' fill='#c6a700'/>
<circle cx='450' cy='258' r='3' fill='#c6a700'/>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>Q16</text>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>QG_16_K_H</text>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>QG_16_K_M</text>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>QG_16_K_L</text>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>Q16_K_L</text>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>Q16_K_M</text>
<circle cx='437' cy='83' r='3' fill='#06c'/>
<text x='442' y='79' fill='#06c'>Q16_K_H</text>
<circle cx='437' cy='138' r='4' fill='#d33' stroke='#900'/>
<text x='442' y='134' fill='#d33'>[ref] IEEE FP16</text>
<circle cx='348' cy='221' r='3' fill='#c6a700'/>
<text x='353' y='217' fill='#c6a700'>QG_MX_12.5</text>
<circle cx='348' cy='233' r='3' fill='#06c'/>
<circle cx='348' cy='233' r='3' fill='#06c'/>
<circle cx='348' cy='233' r='3' fill='#06c'/>
<circle cx='348' cy='233' r='3' fill='#06c'/>
<circle cx='348' cy='233' r='3' fill='#06c'/>
<circle cx='348' cy='233' r='3' fill='#06c'/>
<circle cx='348' cy='270' r='3' fill='#c6a700'/>
<circle cx='335' cy='237' r='3' fill='#06c'/>
<circle cx='335' cy='237' r='3' fill='#06c'/>
<circle cx='335' cy='237' r='3' fill='#06c'/>
<circle cx='335' cy='237' r='3' fill='#06c'/>
<circle cx='246' cy='231' r='3' fill='#06c'/>
<circle cx='246' cy='231' r='3' fill='#06c'/>
<circle cx='246' cy='231' r='3' fill='#06c'/>
<circle cx='246' cy='231' r='3' fill='#06c'/>
<circle cx='246' cy='231' r='3' fill='#06c'/>
<circle cx='246' cy='231' r='3' fill='#06c'/>
<circle cx='246' cy='234' r='4' fill='#d33' stroke='#900'/>
<text x='251' y='230' fill='#d33'>[ref] GGUF Q8_0</text>
<circle cx='246' cy='305' r='3' fill='#c6a700'/>
<circle cx='246' cy='307' r='3' fill='#c6a700'/>
<circle cx='237' cy='239' r='4' fill='#d33' stroke='#900'/>
<text x='242' y='235' fill='#d33'>[ref] INT8 uniform</text>
<circle cx='233' cy='245' r='3' fill='#06c'/>
<circle cx='233' cy='245' r='3' fill='#06c'/>
<circle cx='233' cy='245' r='3' fill='#06c'/>
<circle cx='233' cy='245' r='3' fill='#06c'/>
<circle cx='197' cy='270' r='3' fill='#06c'/>
<circle cx='197' cy='270' r='3' fill='#06c'/>
<circle cx='197' cy='270' r='3' fill='#06c'/>
<circle cx='197' cy='270' r='3' fill='#06c'/>
<circle cx='197' cy='275' r='4' fill='#d33' stroke='#900'/>
<text x='202' y='271' fill='#d33'>[ref] GGUF Q6_K</text>
<circle cx='195' cy='275' r='3' fill='#06c'/>
<circle cx='195' cy='275' r='3' fill='#06c'/>
<circle cx='195' cy='311' r='3' fill='#c6a700'/>
<circle cx='195' cy='316' r='3' fill='#c6a700'/>
<circle cx='183' cy='287' r='3' fill='#06c'/>
<circle cx='183' cy='287' r='3' fill='#06c'/>
<circle cx='183' cy='287' r='3' fill='#06c'/>
<circle cx='183' cy='287' r='3' fill='#06c'/>
<circle cx='144' cy='313' r='3' fill='#06c'/>
<circle cx='144' cy='313' r='3' fill='#06c'/>
<circle cx='144' cy='313' r='3' fill='#06c'/>
<circle cx='144' cy='339' r='3' fill='#c6a700'/>
<circle cx='144' cy='340' r='3' fill='#c6a700'/>
<circle cx='144' cy='373' r='4' fill='#d33' stroke='#900'/>
<text x='149' y='369' fill='#d33'>[ref] GGUF Q4_K</text>
<circle cx='143' cy='313' r='3' fill='#06c'/>
<circle cx='143' cy='313' r='3' fill='#06c'/>
<circle cx='143' cy='313' r='3' fill='#06c'/>
<circle cx='132' cy='324' r='3' fill='#06c'/>
<circle cx='132' cy='324' r='3' fill='#06c'/>
<circle cx='132' cy='324' r='3' fill='#06c'/>
<circle cx='132' cy='324' r='3' fill='#06c'/>
<circle cx='126' cy='368' r='3' fill='#c6a700'/>
<circle cx='119' cy='332' r='3' fill='#06c'/>
<circle cx='119' cy='332' r='3' fill='#06c'/>
<circle cx='119' cy='332' r='3' fill='#06c'/>
<circle cx='119' cy='364' r='3' fill='#c6a700'/>
<circle cx='117' cy='331' r='3' fill='#06c'/>
<circle cx='117' cy='331' r='3' fill='#06c'/>
<circle cx='117' cy='331' r='3' fill='#06c'/>
<circle cx='106' cy='347' r='3' fill='#06c'/>
<circle cx='106' cy='347' r='3' fill='#06c'/>
<circle cx='106' cy='347' r='3' fill='#06c'/>
<circle cx='106' cy='347' r='3' fill='#06c'/>
<circle cx='97' cy='350' r='3' fill='#06c'/>
<circle cx='94' cy='351' r='3' fill='#06c'/>
<circle cx='94' cy='351' r='3' fill='#06c'/>
<circle cx='92' cy='351' r='3' fill='#06c'/>
<circle cx='92' cy='351' r='3' fill='#06c'/>
<circle cx='92' cy='351' r='3' fill='#06c'/>
<circle cx='81' cy='365' r='3' fill='#06c'/>
<circle cx='81' cy='365' r='3' fill='#06c'/>
<circle cx='81' cy='365' r='3' fill='#06c'/>
<circle cx='81' cy='365' r='3' fill='#06c'/>
<circle cx='70' cy='375' r='4' fill='#d33' stroke='#900'/>
<text x='75' y='371' fill='#d33'>[ref] BitNet b1.58</text>
<circle cx='68' cy='371' r='3' fill='#06c'/>
<circle cx='68' cy='371' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='374' r='3' fill='#06c'/>
<circle cx='56' cy='375' r='3' fill='#06c'/>
<circle cx='56' cy='404' r='4' fill='#d33' stroke='#900'/>
<text x='61' y='400' fill='#d33'>[ref] Binary 1-bit</text>
<rect x='640' y='8' width='252' height='52' fill='#fff' stroke='#ccc'/>
<circle cx='654' cy='20' r='3' fill='#06c'/><text x='662' y='23'>Transcender plain</text>
<circle cx='654' cy='34' r='3' fill='#083'/><text x='662' y='37'>Transcender GRP/K_G</text>
<circle cx='654' cy='48' r='3' fill='#c6a700'/><text x='662' y='51'>Transcender MXQ mix</text>
<circle cx='760' cy='20' r='4' fill='#d33' stroke='#900'/><text x='768' y='23'>[ref] industrial</text>
</svg>

| Transcender | BPW | PSNR dB | Competitor | BPW | PSNR dB | Delta | Verdict |
|---|---|---|---|---|---|---|---|
|Q16|16|102.449|[ref] IEEE FP16|16|86.4696|+15.9794 dB|**WIN**|
|QG_MX_16.5|16.5|51.1035|[ref] IEEE FP16|16|86.4696|-35.3661 dB|LOSS|
|QG_8.5|8.5|58.8773|[ref] GGUF Q8_0|8.5|58.1409|+0.7364 dB|**WIN**|
|Q_MX_8.5|8.5|36.5958|[ref] GGUF Q8_0|8.5|58.1409|-21.5451 dB|LOSS|
|QG_6.5|6.5|46.0612|[ref] GGUF Q6_K|6.5625|45.8702|+0.191 dB|**WIN**|
|QG_4.5|4.5|34.8508|[ref] GGUF Q4_K|4.5|17.0351|+17.8157 dB|**WIN**|
|QG_MX_4.5|4.5|27.1477|[ref] GGUF Q4_K|4.5|17.0351|+10.1126 dB|**WIN**|
|QG1|1|16.7458|[ref] BitNet b1.58|1.58|16.3621|+0.3837 dB|**WIN**|
|QG1|1|16.7458|[ref] Binary 1-bit|1|8.10012|+8.64568 dB|**WIN**|
|Q8_K_M|8|54.7039|[ref] INT8 uniform|8.125|56.6672|-1.9633 dB|LOSS|

## Real trained weights — PSNR vs BPW

<svg xmlns='http://www.w3.org/2000/svg' width='900' height='460' font-family='monospace' font-size='10'>
<rect width='100%' height='100%' fill='#fafafa'/>
<line x1='56' y1='404' x2='844' y2='404' stroke='#ddd'/>
<text x='4' y='407'>10 dB</text>
<text x='44' y='418'>1 bpw</text>
<line x1='56' y1='317' x2='844' y2='317' stroke='#ddd'/>
<text x='4' y='320'>35 dB</text>
<text x='241' y='418'>8.75 bpw</text>
<line x1='56' y1='230' x2='844' y2='230' stroke='#ddd'/>
<text x='4' y='233'>60 dB</text>
<text x='438' y='418'>16.5 bpw</text>
<line x1='56' y1='143' x2='844' y2='143' stroke='#ddd'/>
<text x='4' y='146'>85 dB</text>
<text x='635' y='418'>24.25 bpw</text>
<line x1='56' y1='56' x2='844' y2='56' stroke='#ddd'/>
<text x='4' y='59'>110 dB</text>
<text x='832' y='418'>32 bpw</text>
<line x1='56' y1='404' x2='844' y2='404' stroke='#333'/>
<line x1='56' y1='28' x2='56' y2='404' stroke='#333'/>
<circle cx='844' cy='92' r='3' fill='#06c'/>
<text x='849' y='88' fill='#06c'>Q32</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>QG24</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>QG_24.5</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>Q24.5</text>
<circle cx='653' cy='231' r='3' fill='#c6a700'/>
<circle cx='653' cy='231' r='3' fill='#c6a700'/>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>QG_24_K_H</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>QG_24_K_M</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>QG_24_K_L</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_L</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_M</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_H</text>
<circle cx='450' cy='76' r='3' fill='#06c'/>
<text x='455' y='72' fill='#06c'>QG16</text>
<circle cx='450' cy='77' r='3' fill='#06c'/>
<text x='455' y='73' fill='#06c'>Q16.5</text>
<circle cx='450' cy='77' r='3' fill='#06c'/>
<text x='455' y='73' fill='#06c'>QG_16.5</text>
<circle cx='450' cy='253' r='3' fill='#c6a700'/>
<circle cx='450' cy='268' r='3' fill='#c6a700'/>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>Q16</text>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>QG_16_K_H</text>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>QG_16_K_M</text>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>QG_16_K_L</text>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>Q16_K_L</text>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>Q16_K_M</text>
<circle cx='437' cy='77' r='3' fill='#06c'/>
<text x='442' y='73' fill='#06c'>Q16_K_H</text>
<circle cx='437' cy='140' r='4' fill='#d33' stroke='#900'/>
<text x='442' y='136' fill='#d33'>[ref] IEEE FP16</text>
<circle cx='348' cy='226' r='3' fill='#c6a700'/>
<text x='353' y='222' fill='#c6a700'>QG_MX_12.5</text>
<circle cx='348' cy='252' r='3' fill='#06c'/>
<circle cx='348' cy='271' r='3' fill='#06c'/>
<circle cx='348' cy='271' r='3' fill='#06c'/>
<circle cx='348' cy='271' r='3' fill='#06c'/>
<circle cx='348' cy='271' r='3' fill='#06c'/>
<circle cx='348' cy='271' r='3' fill='#06c'/>
<circle cx='348' cy='273' r='3' fill='#c6a700'/>
<circle cx='335' cy='272' r='3' fill='#06c'/>
<circle cx='335' cy='272' r='3' fill='#06c'/>
<circle cx='335' cy='272' r='3' fill='#06c'/>
<circle cx='335' cy='272' r='3' fill='#06c'/>
<circle cx='246' cy='230' r='3' fill='#06c'/>
<circle cx='246' cy='230' r='3' fill='#06c'/>
<circle cx='246' cy='230' r='3' fill='#06c'/>
<circle cx='246' cy='232' r='4' fill='#d33' stroke='#900'/>
<text x='251' y='228' fill='#d33'>[ref] GGUF Q8_0</text>
<circle cx='246' cy='254' r='3' fill='#06c'/>
<circle cx='246' cy='254' r='3' fill='#06c'/>
<circle cx='246' cy='254' r='3' fill='#06c'/>
<circle cx='246' cy='313' r='3' fill='#c6a700'/>
<circle cx='246' cy='314' r='3' fill='#c6a700'/>
<circle cx='237' cy='236' r='4' fill='#d33' stroke='#900'/>
<text x='242' y='232' fill='#d33'>[ref] INT8 uniform</text>
<circle cx='233' cy='241' r='3' fill='#06c'/>
<circle cx='233' cy='241' r='3' fill='#06c'/>
<circle cx='233' cy='241' r='3' fill='#06c'/>
<circle cx='233' cy='241' r='3' fill='#06c'/>
<circle cx='197' cy='271' r='3' fill='#06c'/>
<circle cx='197' cy='271' r='3' fill='#06c'/>
<circle cx='197' cy='271' r='3' fill='#06c'/>
<circle cx='197' cy='271' r='3' fill='#06c'/>
<circle cx='197' cy='280' r='4' fill='#d33' stroke='#900'/>
<text x='202' y='276' fill='#d33'>[ref] GGUF Q6_K</text>
<circle cx='195' cy='274' r='3' fill='#06c'/>
<circle cx='195' cy='274' r='3' fill='#06c'/>
<circle cx='195' cy='317' r='3' fill='#c6a700'/>
<circle cx='195' cy='321' r='3' fill='#c6a700'/>
<circle cx='183' cy='306' r='3' fill='#06c'/>
<circle cx='183' cy='306' r='3' fill='#06c'/>
<circle cx='183' cy='306' r='3' fill='#06c'/>
<circle cx='183' cy='306' r='3' fill='#06c'/>
<circle cx='144' cy='314' r='3' fill='#06c'/>
<circle cx='144' cy='314' r='3' fill='#06c'/>
<circle cx='144' cy='314' r='3' fill='#06c'/>
<circle cx='144' cy='350' r='3' fill='#c6a700'/>
<circle cx='144' cy='351' r='3' fill='#c6a700'/>
<circle cx='144' cy='376' r='4' fill='#d33' stroke='#900'/>
<text x='149' y='372' fill='#d33'>[ref] GGUF Q4_K</text>
<circle cx='143' cy='318' r='3' fill='#06c'/>
<circle cx='143' cy='318' r='3' fill='#06c'/>
<circle cx='143' cy='318' r='3' fill='#06c'/>
<circle cx='132' cy='329' r='3' fill='#06c'/>
<circle cx='132' cy='329' r='3' fill='#06c'/>
<circle cx='132' cy='329' r='3' fill='#06c'/>
<circle cx='132' cy='329' r='3' fill='#06c'/>
<circle cx='126' cy='374' r='3' fill='#c6a700'/>
<circle cx='119' cy='334' r='3' fill='#06c'/>
<circle cx='119' cy='334' r='3' fill='#06c'/>
<circle cx='119' cy='334' r='3' fill='#06c'/>
<circle cx='119' cy='373' r='3' fill='#c6a700'/>
<circle cx='117' cy='334' r='3' fill='#06c'/>
<circle cx='117' cy='334' r='3' fill='#06c'/>
<circle cx='117' cy='334' r='3' fill='#06c'/>
<circle cx='106' cy='348' r='3' fill='#06c'/>
<circle cx='106' cy='348' r='3' fill='#06c'/>
<circle cx='106' cy='348' r='3' fill='#06c'/>
<circle cx='106' cy='348' r='3' fill='#06c'/>
<circle cx='97' cy='353' r='3' fill='#06c'/>
<circle cx='94' cy='354' r='3' fill='#06c'/>
<circle cx='94' cy='354' r='3' fill='#06c'/>
<circle cx='92' cy='354' r='3' fill='#06c'/>
<circle cx='92' cy='354' r='3' fill='#06c'/>
<circle cx='92' cy='354' r='3' fill='#06c'/>
<circle cx='81' cy='368' r='3' fill='#06c'/>
<circle cx='81' cy='368' r='3' fill='#06c'/>
<circle cx='81' cy='368' r='3' fill='#06c'/>
<circle cx='81' cy='368' r='3' fill='#06c'/>
<circle cx='70' cy='379' r='4' fill='#d33' stroke='#900'/>
<text x='75' y='375' fill='#d33'>[ref] BitNet b1.58</text>
<circle cx='68' cy='376' r='3' fill='#06c'/>
<circle cx='68' cy='376' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='381' r='3' fill='#06c'/>
<circle cx='56' cy='382' r='3' fill='#06c'/>
<circle cx='56' cy='404' r='4' fill='#d33' stroke='#900'/>
<text x='61' y='400' fill='#d33'>[ref] Binary 1-bit</text>
<rect x='640' y='8' width='252' height='52' fill='#fff' stroke='#ccc'/>
<circle cx='654' cy='20' r='3' fill='#06c'/><text x='662' y='23'>Transcender plain</text>
<circle cx='654' cy='34' r='3' fill='#083'/><text x='662' y='37'>Transcender GRP/K_G</text>
<circle cx='654' cy='48' r='3' fill='#c6a700'/><text x='662' y='51'>Transcender MXQ mix</text>
<circle cx='760' cy='20' r='4' fill='#d33' stroke='#900'/><text x='768' y='23'>[ref] industrial</text>
</svg>

| Transcender | BPW | PSNR dB | Competitor | BPW | PSNR dB | Delta | Verdict |
|---|---|---|---|---|---|---|---|
|Q16|16|104.511|[ref] IEEE FP16|16|86.2819|+18.2291 dB|**WIN**|
|QG_MX_16.5|16.5|49.5143|[ref] IEEE FP16|16|86.2819|-36.7676 dB|LOSS|
|QG_8.5|8.5|60.3689|[ref] GGUF Q8_0|8.5|59.7489|+0.62 dB|**WIN**|
|Q_MX_8.5|8.5|36.1751|[ref] GGUF Q8_0|8.5|59.7489|-23.5738 dB|LOSS|
|QG_6.5|6.5|47.5979|[ref] GGUF Q6_K|6.5625|46.1225|+1.4754 dB|**WIN**|
|QG_4.5|4.5|36.1389|[ref] GGUF Q4_K|4.5|18.3321|+17.8068 dB|**WIN**|
|QG_MX_4.5|4.5|25.7821|[ref] GGUF Q4_K|4.5|18.3321|+7.45 dB|**WIN**|
|QG1|1|16.9532|[ref] BitNet b1.58|1.58|17.6669|-0.7137 dB|LOSS|
|QG1|1|16.9532|[ref] Binary 1-bit|1|10.4861|+6.4671 dB|**WIN**|
|Q8_K_M|8|57.0857|[ref] INT8 uniform|8.125|58.7437|-1.658 dB|LOSS|

## Competitor landscape

See `docs/COMPETITOR_ANALYSIS.md` for the researched mapping
(GGUF K-quants/IQ, GPTQ, AWQ, SmoothQuant, SpQR, SqueezeLLM, AQLM,
QuIP#, EXL2/EXL3, BitNet b1.58, BinaryNet) with their published metrics
and honest notes on metric differences (weight-PSNR vs perplexity).

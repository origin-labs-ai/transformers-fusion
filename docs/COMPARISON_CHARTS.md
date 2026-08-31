# InNova vs Industrial Baselines — REAL Measured Charts

> Generated from `bench_format_comparison.csv` by `tools/generate_comparison_visuals.cpp`.
> Every number below comes from a measured round-trip of the production codec.
> BPW ironclad: each format stores EXACTLY the BPW in its name.

## Gaussian weights — PSNR vs BPW

<svg xmlns='http://www.w3.org/2000/svg' width='900' height='460' font-family='monospace' font-size='10'>
<rect width='100%' height='100%' fill='#fafafa'/>
<line x1='56' y1='404' x2='844' y2='404' stroke='#ddd'/>
<text x='4' y='407'>-30 dB</text>
<text x='44' y='418'>1 bpw</text>
<line x1='56' y1='317' x2='844' y2='317' stroke='#ddd'/>
<text x='4' y='320'>4 dB</text>
<text x='241' y='418'>8.75 bpw</text>
<line x1='56' y1='230' x2='844' y2='230' stroke='#ddd'/>
<text x='4' y='233'>40 dB</text>
<text x='438' y='418'>16.5 bpw</text>
<line x1='56' y1='143' x2='844' y2='143' stroke='#ddd'/>
<text x='4' y='146'>75 dB</text>
<text x='635' y='418'>24.25 bpw</text>
<line x1='56' y1='56' x2='844' y2='56' stroke='#ddd'/>
<text x='4' y='59'>110 dB</text>
<text x='832' y='418'>32 bpw</text>
<line x1='56' y1='404' x2='844' y2='404' stroke='#333'/>
<line x1='56' y1='28' x2='56' y2='404' stroke='#333'/>
<circle cx='844' cy='82' r='3' fill='#06c'/>
<text x='849' y='78' fill='#06c'>Q32</text>
<circle cx='653' cy='56' r='3' fill='#083'/>
<text x='658' y='52' fill='#083'>Q_G_24.5</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>Q24.5</text>
<circle cx='653' cy='177' r='3' fill='#c6a700'/>
<text x='658' y='173' fill='#c6a700'>MXQ_24.5</text>
<circle cx='653' cy='177' r='3' fill='#c6a700'/>
<text x='658' y='173' fill='#c6a700'>MXQ_24.5_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_K_H_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_K_M_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_K_L_G</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_L</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_M</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_H</text>
<circle cx='450' cy='76' r='3' fill='#06c'/>
<text x='455' y='72' fill='#06c'>Q16.5</text>
<circle cx='450' cy='76' r='3' fill='#083'/>
<text x='455' y='72' fill='#083'>Q_G_16.5</text>
<circle cx='450' cy='192' r='3' fill='#c6a700'/>
<circle cx='450' cy='203' r='3' fill='#c6a700'/>
<circle cx='437' cy='75' r='3' fill='#083'/>
<text x='442' y='71' fill='#083'>Q16_G</text>
<circle cx='437' cy='76' r='3' fill='#06c'/>
<text x='442' y='72' fill='#06c'>Q16</text>
<circle cx='437' cy='76' r='3' fill='#083'/>
<text x='442' y='72' fill='#083'>Q16_K_H_G</text>
<circle cx='437' cy='76' r='3' fill='#083'/>
<text x='442' y='72' fill='#083'>Q16_K_M_G</text>
<circle cx='437' cy='76' r='3' fill='#083'/>
<text x='442' y='72' fill='#083'>Q16_K_L_G</text>
<circle cx='437' cy='76' r='3' fill='#06c'/>
<text x='442' y='72' fill='#06c'>Q16_K_L</text>
<circle cx='437' cy='76' r='3' fill='#06c'/>
<text x='442' y='72' fill='#06c'>Q16_K_M</text>
<circle cx='437' cy='76' r='3' fill='#06c'/>
<text x='442' y='72' fill='#06c'>Q16_K_H</text>
<circle cx='437' cy='115' r='4' fill='#d33' stroke='#900'/>
<text x='442' y='111' fill='#d33'>[ref] IEEE FP16</text>
<circle cx='348' cy='185' r='3' fill='#06c'/>
<circle cx='348' cy='185' r='3' fill='#083'/>
<circle cx='348' cy='212' r='3' fill='#c6a700'/>
<circle cx='348' cy='213' r='3' fill='#c6a700'/>
<circle cx='335' cy='185' r='3' fill='#083'/>
<circle cx='335' cy='185' r='3' fill='#083'/>
<circle cx='335' cy='185' r='3' fill='#083'/>
<circle cx='335' cy='185' r='3' fill='#083'/>
<circle cx='335' cy='187' r='3' fill='#06c'/>
<circle cx='335' cy='187' r='3' fill='#06c'/>
<circle cx='335' cy='187' r='3' fill='#06c'/>
<circle cx='335' cy='187' r='3' fill='#06c'/>
<circle cx='246' cy='184' r='3' fill='#06c'/>
<circle cx='246' cy='184' r='3' fill='#083'/>
<circle cx='246' cy='185' r='4' fill='#d33' stroke='#900'/>
<text x='251' y='181' fill='#d33'>[ref] GGUF Q8_0</text>
<circle cx='246' cy='238' r='3' fill='#c6a700'/>
<circle cx='246' cy='240' r='3' fill='#c6a700'/>
<circle cx='237' cy='189' r='4' fill='#d33' stroke='#900'/>
<text x='242' y='185' fill='#d33'>[ref] INT8 uniform</text>
<circle cx='233' cy='184' r='3' fill='#083'/>
<circle cx='233' cy='184' r='3' fill='#083'/>
<circle cx='233' cy='184' r='3' fill='#083'/>
<circle cx='233' cy='184' r='3' fill='#083'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='197' cy='215' r='4' fill='#d33' stroke='#900'/>
<text x='202' y='211' fill='#d33'>[ref] GGUF Q6_K</text>
<circle cx='195' cy='215' r='3' fill='#06c'/>
<circle cx='195' cy='215' r='3' fill='#083'/>
<circle cx='195' cy='245' r='3' fill='#c6a700'/>
<circle cx='195' cy='248' r='3' fill='#c6a700'/>
<circle cx='183' cy='212' r='3' fill='#083'/>
<circle cx='183' cy='224' r='3' fill='#06c'/>
<circle cx='183' cy='224' r='3' fill='#06c'/>
<circle cx='183' cy='224' r='3' fill='#06c'/>
<circle cx='183' cy='224' r='3' fill='#06c'/>
<circle cx='183' cy='404' r='3' fill='#083'/>
<circle cx='183' cy='404' r='3' fill='#083'/>
<circle cx='183' cy='404' r='3' fill='#083'/>
<circle cx='144' cy='243' r='3' fill='#083'/>
<circle cx='144' cy='243' r='3' fill='#06c'/>
<circle cx='144' cy='262' r='3' fill='#c6a700'/>
<circle cx='144' cy='285' r='3' fill='#c6a700'/>
<circle cx='144' cy='287' r='4' fill='#d33' stroke='#900'/>
<text x='149' y='283' fill='#d33'>[ref] GGUF Q4_K</text>
<circle cx='132' cy='243' r='3' fill='#083'/>
<circle cx='132' cy='243' r='3' fill='#083'/>
<circle cx='132' cy='243' r='3' fill='#083'/>
<circle cx='132' cy='243' r='3' fill='#083'/>
<circle cx='132' cy='251' r='3' fill='#06c'/>
<circle cx='132' cy='251' r='3' fill='#06c'/>
<circle cx='132' cy='251' r='3' fill='#06c'/>
<circle cx='132' cy='251' r='3' fill='#06c'/>
<circle cx='119' cy='257' r='3' fill='#06c'/>
<circle cx='119' cy='257' r='3' fill='#083'/>
<circle cx='119' cy='280' r='3' fill='#c6a700'/>
<circle cx='119' cy='285' r='3' fill='#c6a700'/>
<circle cx='106' cy='256' r='3' fill='#083'/>
<circle cx='106' cy='256' r='3' fill='#083'/>
<circle cx='106' cy='256' r='3' fill='#083'/>
<circle cx='106' cy='257' r='3' fill='#083'/>
<circle cx='106' cy='268' r='3' fill='#06c'/>
<circle cx='106' cy='268' r='3' fill='#06c'/>
<circle cx='106' cy='268' r='3' fill='#06c'/>
<circle cx='106' cy='268' r='3' fill='#06c'/>
<circle cx='94' cy='270' r='3' fill='#083'/>
<circle cx='94' cy='270' r='3' fill='#06c'/>
<circle cx='81' cy='270' r='3' fill='#083'/>
<circle cx='81' cy='270' r='3' fill='#083'/>
<circle cx='81' cy='270' r='3' fill='#083'/>
<circle cx='81' cy='270' r='3' fill='#083'/>
<circle cx='81' cy='281' r='3' fill='#06c'/>
<circle cx='81' cy='281' r='3' fill='#06c'/>
<circle cx='81' cy='281' r='3' fill='#06c'/>
<circle cx='81' cy='281' r='3' fill='#06c'/>
<circle cx='70' cy='288' r='4' fill='#d33' stroke='#900'/>
<text x='75' y='284' fill='#d33'>[ref] BitNet b1.58</text>
<circle cx='68' cy='298' r='3' fill='#083'/>
<circle cx='68' cy='298' r='3' fill='#06c'/>
<circle cx='56' cy='287' r='3' fill='#06c'/>
<circle cx='56' cy='287' r='3' fill='#083'/>
<circle cx='56' cy='287' r='3' fill='#083'/>
<circle cx='56' cy='287' r='3' fill='#083'/>
<circle cx='56' cy='287' r='3' fill='#083'/>
<circle cx='56' cy='287' r='3' fill='#06c'/>
<circle cx='56' cy='287' r='3' fill='#06c'/>
<circle cx='56' cy='288' r='3' fill='#06c'/>
<circle cx='56' cy='309' r='4' fill='#d33' stroke='#900'/>
<text x='61' y='305' fill='#d33'>[ref] Binary 1-bit</text>
<rect x='640' y='8' width='252' height='52' fill='#fff' stroke='#ccc'/>
<circle cx='654' cy='20' r='3' fill='#06c'/><text x='662' y='23'>InNova plain</text>
<circle cx='654' cy='34' r='3' fill='#083'/><text x='662' y='37'>InNova GRP/K_G</text>
<circle cx='654' cy='48' r='3' fill='#c6a700'/><text x='662' y='51'>InNova MXQ mix</text>
<circle cx='760' cy='20' r='4' fill='#d33' stroke='#900'/><text x='768' y='23'>[ref] industrial</text>
</svg>

| InNova | BPW | PSNR dB | Competitor | BPW | PSNR dB | Delta | Verdict |
|---|---|---|---|---|---|---|---|
|Q16|16|102.449|[ref] IEEE FP16|16|86.4696|+15.9794 dB|**WIN**|
|MXQ_16.5_G|16.5|51.1035|[ref] IEEE FP16|16|86.4696|-35.3661 dB|LOSS|
|Q_G_8.5|8.5|58.5633|[ref] GGUF Q8_0|8.5|58.1409|+0.4224 dB|**WIN**|
|MXQ_8.5|8.5|36.5958|[ref] GGUF Q8_0|8.5|58.1409|-21.5451 dB|LOSS|
|Q_G_6.5|6.5|46.0612|[ref] GGUF Q6_K|6.5625|45.8702|+0.191 dB|**WIN**|
|Q_G_4.5|4.5|34.8508|[ref] GGUF Q4_K|4.5|17.0351|+17.8157 dB|**WIN**|
|MXQ_4.5_G|4.5|17.8036|[ref] GGUF Q4_K|4.5|17.0351|+0.7685 dB|**WIN**|
|Q1_G|1|16.7458|[ref] BitNet b1.58|1.58|16.3621|+0.3837 dB|**WIN**|
|Q1_G|1|16.7458|[ref] Binary 1-bit|1|8.10012|+8.64568 dB|**WIN**|
|Q8_K_M|8|54.7213|[ref] INT8 uniform|8.125|56.6672|-1.9459 dB|LOSS|

## Real trained weights — PSNR vs BPW

<svg xmlns='http://www.w3.org/2000/svg' width='900' height='460' font-family='monospace' font-size='10'>
<rect width='100%' height='100%' fill='#fafafa'/>
<line x1='56' y1='404' x2='844' y2='404' stroke='#ddd'/>
<text x='4' y='407'>-23 dB</text>
<text x='44' y='418'>1 bpw</text>
<line x1='56' y1='317' x2='844' y2='317' stroke='#ddd'/>
<text x='4' y='320'>9 dB</text>
<text x='241' y='418'>8.75 bpw</text>
<line x1='56' y1='230' x2='844' y2='230' stroke='#ddd'/>
<text x='4' y='233'>43 dB</text>
<text x='438' y='418'>16.5 bpw</text>
<line x1='56' y1='143' x2='844' y2='143' stroke='#ddd'/>
<text x='4' y='146'>77 dB</text>
<text x='635' y='418'>24.25 bpw</text>
<line x1='56' y1='56' x2='844' y2='56' stroke='#ddd'/>
<text x='4' y='59'>110 dB</text>
<text x='832' y='418'>32 bpw</text>
<line x1='56' y1='404' x2='844' y2='404' stroke='#333'/>
<line x1='56' y1='28' x2='56' y2='404' stroke='#333'/>
<circle cx='844' cy='83' r='3' fill='#06c'/>
<text x='849' y='79' fill='#06c'>Q32</text>
<circle cx='653' cy='56' r='3' fill='#083'/>
<text x='658' y='52' fill='#083'>Q_G_24.5</text>
<circle cx='653' cy='56' r='3' fill='#06c'/>
<text x='658' y='52' fill='#06c'>Q24.5</text>
<circle cx='653' cy='186' r='3' fill='#c6a700'/>
<circle cx='653' cy='186' r='3' fill='#c6a700'/>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_K_H_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_K_M_G</text>
<circle cx='640' cy='56' r='3' fill='#083'/>
<text x='645' y='52' fill='#083'>Q24_K_L_G</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_L</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_M</text>
<circle cx='640' cy='56' r='3' fill='#06c'/>
<text x='645' y='52' fill='#06c'>Q24_K_H</text>
<circle cx='450' cy='71' r='3' fill='#06c'/>
<text x='455' y='67' fill='#06c'>Q16.5</text>
<circle cx='450' cy='71' r='3' fill='#083'/>
<text x='455' y='67' fill='#083'>Q_G_16.5</text>
<circle cx='450' cy='203' r='3' fill='#c6a700'/>
<circle cx='450' cy='214' r='3' fill='#c6a700'/>
<circle cx='437' cy='71' r='3' fill='#083'/>
<text x='442' y='67' fill='#083'>Q16_G</text>
<circle cx='437' cy='71' r='3' fill='#06c'/>
<text x='442' y='67' fill='#06c'>Q16</text>
<circle cx='437' cy='71' r='3' fill='#083'/>
<text x='442' y='67' fill='#083'>Q16_K_H_G</text>
<circle cx='437' cy='71' r='3' fill='#083'/>
<text x='442' y='67' fill='#083'>Q16_K_M_G</text>
<circle cx='437' cy='71' r='3' fill='#083'/>
<text x='442' y='67' fill='#083'>Q16_K_L_G</text>
<circle cx='437' cy='71' r='3' fill='#06c'/>
<text x='442' y='67' fill='#06c'>Q16_K_L</text>
<circle cx='437' cy='71' r='3' fill='#06c'/>
<text x='442' y='67' fill='#06c'>Q16_K_M</text>
<circle cx='437' cy='71' r='3' fill='#06c'/>
<text x='442' y='67' fill='#06c'>Q16_K_H</text>
<circle cx='437' cy='119' r='4' fill='#d33' stroke='#900'/>
<text x='442' y='115' fill='#d33'>[ref] IEEE FP16</text>
<circle cx='348' cy='216' r='3' fill='#06c'/>
<circle cx='348' cy='216' r='3' fill='#083'/>
<circle cx='348' cy='218' r='3' fill='#c6a700'/>
<circle cx='348' cy='219' r='3' fill='#c6a700'/>
<circle cx='335' cy='202' r='3' fill='#083'/>
<circle cx='335' cy='216' r='3' fill='#083'/>
<circle cx='335' cy='216' r='3' fill='#083'/>
<circle cx='335' cy='216' r='3' fill='#083'/>
<circle cx='335' cy='217' r='3' fill='#06c'/>
<circle cx='335' cy='217' r='3' fill='#06c'/>
<circle cx='335' cy='217' r='3' fill='#06c'/>
<circle cx='335' cy='217' r='3' fill='#06c'/>
<circle cx='246' cy='186' r='3' fill='#06c'/>
<circle cx='246' cy='186' r='3' fill='#083'/>
<circle cx='246' cy='187' r='4' fill='#d33' stroke='#900'/>
<text x='251' y='183' fill='#d33'>[ref] GGUF Q8_0</text>
<circle cx='246' cy='248' r='3' fill='#c6a700'/>
<circle cx='246' cy='250' r='3' fill='#c6a700'/>
<circle cx='237' cy='190' r='4' fill='#d33' stroke='#900'/>
<text x='242' y='186' fill='#d33'>[ref] INT8 uniform</text>
<circle cx='233' cy='186' r='3' fill='#083'/>
<circle cx='233' cy='186' r='3' fill='#083'/>
<circle cx='233' cy='186' r='3' fill='#083'/>
<circle cx='233' cy='186' r='3' fill='#083'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='233' cy='194' r='3' fill='#06c'/>
<circle cx='197' cy='223' r='4' fill='#d33' stroke='#900'/>
<text x='202' y='219' fill='#d33'>[ref] GGUF Q6_K</text>
<circle cx='195' cy='219' r='3' fill='#06c'/>
<circle cx='195' cy='219' r='3' fill='#083'/>
<circle cx='195' cy='253' r='3' fill='#c6a700'/>
<circle cx='195' cy='255' r='3' fill='#c6a700'/>
<circle cx='183' cy='216' r='3' fill='#083'/>
<circle cx='183' cy='242' r='3' fill='#06c'/>
<circle cx='183' cy='242' r='3' fill='#06c'/>
<circle cx='183' cy='242' r='3' fill='#06c'/>
<circle cx='183' cy='242' r='3' fill='#06c'/>
<circle cx='183' cy='404' r='3' fill='#083'/>
<circle cx='183' cy='404' r='3' fill='#083'/>
<circle cx='183' cy='404' r='3' fill='#083'/>
<circle cx='144' cy='248' r='3' fill='#083'/>
<circle cx='144' cy='248' r='3' fill='#06c'/>
<circle cx='144' cy='276' r='3' fill='#c6a700'/>
<circle cx='144' cy='291' r='3' fill='#c6a700'/>
<circle cx='144' cy='294' r='4' fill='#d33' stroke='#900'/>
<text x='149' y='290' fill='#d33'>[ref] GGUF Q4_K</text>
<circle cx='132' cy='248' r='3' fill='#083'/>
<circle cx='132' cy='251' r='3' fill='#083'/>
<circle cx='132' cy='251' r='3' fill='#083'/>
<circle cx='132' cy='251' r='3' fill='#083'/>
<circle cx='132' cy='259' r='3' fill='#06c'/>
<circle cx='132' cy='259' r='3' fill='#06c'/>
<circle cx='132' cy='259' r='3' fill='#06c'/>
<circle cx='132' cy='259' r='3' fill='#06c'/>
<circle cx='119' cy='263' r='3' fill='#06c'/>
<circle cx='119' cy='263' r='3' fill='#083'/>
<circle cx='119' cy='292' r='3' fill='#c6a700'/>
<circle cx='119' cy='296' r='3' fill='#c6a700'/>
<circle cx='106' cy='263' r='3' fill='#083'/>
<circle cx='106' cy='263' r='3' fill='#083'/>
<circle cx='106' cy='263' r='3' fill='#083'/>
<circle cx='106' cy='263' r='3' fill='#083'/>
<circle cx='106' cy='273' r='3' fill='#06c'/>
<circle cx='106' cy='273' r='3' fill='#06c'/>
<circle cx='106' cy='273' r='3' fill='#06c'/>
<circle cx='106' cy='273' r='3' fill='#06c'/>
<circle cx='94' cy='278' r='3' fill='#083'/>
<circle cx='94' cy='278' r='3' fill='#06c'/>
<circle cx='81' cy='277' r='3' fill='#083'/>
<circle cx='81' cy='278' r='3' fill='#083'/>
<circle cx='81' cy='278' r='3' fill='#083'/>
<circle cx='81' cy='278' r='3' fill='#083'/>
<circle cx='81' cy='288' r='3' fill='#06c'/>
<circle cx='81' cy='288' r='3' fill='#06c'/>
<circle cx='81' cy='288' r='3' fill='#06c'/>
<circle cx='81' cy='288' r='3' fill='#06c'/>
<circle cx='70' cy='296' r='4' fill='#d33' stroke='#900'/>
<text x='75' y='292' fill='#d33'>[ref] BitNet b1.58</text>
<circle cx='68' cy='310' r='3' fill='#083'/>
<circle cx='68' cy='310' r='3' fill='#06c'/>
<circle cx='56' cy='298' r='3' fill='#06c'/>
<circle cx='56' cy='298' r='3' fill='#083'/>
<circle cx='56' cy='298' r='3' fill='#083'/>
<circle cx='56' cy='298' r='3' fill='#083'/>
<circle cx='56' cy='298' r='3' fill='#083'/>
<circle cx='56' cy='298' r='3' fill='#06c'/>
<circle cx='56' cy='298' r='3' fill='#06c'/>
<circle cx='56' cy='299' r='3' fill='#06c'/>
<circle cx='56' cy='315' r='4' fill='#d33' stroke='#900'/>
<text x='61' y='311' fill='#d33'>[ref] Binary 1-bit</text>
<rect x='640' y='8' width='252' height='52' fill='#fff' stroke='#ccc'/>
<circle cx='654' cy='20' r='3' fill='#06c'/><text x='662' y='23'>InNova plain</text>
<circle cx='654' cy='34' r='3' fill='#083'/><text x='662' y='37'>InNova GRP/K_G</text>
<circle cx='654' cy='48' r='3' fill='#c6a700'/><text x='662' y='51'>InNova MXQ mix</text>
<circle cx='760' cy='20' r='4' fill='#d33' stroke='#900'/><text x='768' y='23'>[ref] industrial</text>
</svg>

| InNova | BPW | PSNR dB | Competitor | BPW | PSNR dB | Delta | Verdict |
|---|---|---|---|---|---|---|---|
|Q16|16|104.504|[ref] IEEE FP16|16|86.283|+18.221 dB|**WIN**|
|MXQ_16.5_G|16.5|49.5144|[ref] IEEE FP16|16|86.283|-36.7686 dB|LOSS|
|Q_G_8.5|8.5|60.1462|[ref] GGUF Q8_0|8.5|59.7495|+0.3967 dB|**WIN**|
|MXQ_8.5|8.5|36.1751|[ref] GGUF Q8_0|8.5|59.7495|-23.5744 dB|LOSS|
|Q_G_6.5|6.5|47.5974|[ref] GGUF Q6_K|6.5625|46.1225|+1.4749 dB|**WIN**|
|Q_G_4.5|4.5|36.1331|[ref] GGUF Q4_K|4.5|18.3321|+17.801 dB|**WIN**|
|MXQ_4.5_G|4.5|19.5346|[ref] GGUF Q4_K|4.5|18.3321|+1.2025 dB|**WIN**|
|Q1_G|1|16.9532|[ref] BitNet b1.58|1.58|17.667|-0.7138 dB|LOSS|
|Q1_G|1|16.9532|[ref] Binary 1-bit|1|10.4861|+6.4671 dB|**WIN**|
|Q8_K_M|8|57.0714|[ref] INT8 uniform|8.125|58.7437|-1.6723 dB|LOSS|

## Competitor landscape

See `docs/COMPETITOR_ANALYSIS.md` for the researched mapping
(GGUF K-quants/IQ, GPTQ, AWQ, SmoothQuant, SpQR, SqueezeLLM, AQLM,
QuIP#, EXL2/EXL3, BitNet b1.58, BinaryNet) with their published metrics
and honest notes on metric differences (weight-PSNR vs perplexity).

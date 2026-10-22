# ROB1 2025 semestral work hardware files

Assignment details: https://cw.fel.cvut.cz/wiki/courses/b3b33rob1/final_project
Documentation: https://robotics-robot-stations.readthedocs.io/en/latest/index.html


## TODO
- [X] Základní seznámení 15.10
- [ ] Kalibrace kamery pomocí různých pozic kroužku

## High-level dokumentace kódu

### camera.py

1) Funkce take_calibration_pictures zařizují získání obrázku kroužku/gripperu v různých bodech různě vysokých rovin. Pozice a rotace gripperu v každém obrázku se ukládájí jako metadatový JSON do METADATA_PATH. Data z JSONu lze jednoduše zpětně načíst pomocí funkce load_calibration_data.
2) Funkce detect_hoop_positions z obrázků z předchozího kroku identifikuje kroužek pomocí hue-distance a kulatosti objektu (chyba kulatosti je $|\frac{4 \pi A}{o^2 + \epsilon} - 1|$).
3) Funkce calibrate_camera využívá algoritmu DLT pro nalezení projekční matice $P$. Ta je poté dekomponována na matici kamery $K$, rotační matici mezi kamerou a základnou a translační vektor mezi kamerou a základnou (základna = (0,0,0)). **Implementace AI**
4) Funkce estimate_aruco_world_positions poté z fotky otáčivého stolku detekuje rotaci a translaci detekovaných ArUco markerů (známé velikosti) v globálních souřadnicích. Opakováným voláním v čase bychom měli být schopni určit úhel a úhlovou rychlost otáčení, včetně změny směru. Je možné, že tato funkce bude do zpětné vazby dodávata relativně velké **zpoždění**, takže ji možná bude třeba do budoucna zoptimalizovat! **Implementace AI**


#### Dodatek: možné problémy

Je možné, že hledání kalibrační matice $K$ z bodů gripperu nebude dostatečně přesné - pak by bylo třeba $K$ najít nezávislým experimentem s šachovnicí.

Pro zvýšení počtu testovacích bodů by teoreticky bylo možné brát v potaz i rozměry kroužku a neuvažovat jenom střed kroužku, ale také body nad, pod, vlevo a vpravo od středu s pousunutým vektorem pozice v rovině `X-Y` or $(\pm R,\pm R)$. Otázkou je jestli je nelinearita kamery dosttatečně zanedbatelná pro tuto operaci.

#### Dodatek: DLT
DLT (Direct linear transformation) je algoritmus na hledání projekční matice, která nejlépe dopovídá našim korespondencím. Ta se najde sestavením rovnic pro každou korespondenci:
$$
  \begin{align}
    \lambda_i \begin{bmatrix}
           u_i \\
           v_i \\
           1 \\
         \end{bmatrix} = P^{3 \times 4} \begin{bmatrix}
           x_i \\
           y_i \\
           z_i \\
           1 \\
         \end{bmatrix}= \begin{bmatrix}
            p_{11} & p_{12} & p_{13} & p_{14}\\
            p_{21} & p_{22} & p_{23} & p_{24}\\
            p_{31} & p_{32} & p_{33} & p_{34}\\
         \end{bmatrix} \begin{bmatrix}
           x_i \\
           y_i \\
           z_i \\
           1 \\
         \end{bmatrix}
  \end{align}
$$

Po přepisu:
$$
  \begin{align}
    \lambda_i u_i = p_{11}x_i + p_{12}y_i + p_{13}z_i + p_{14}\\
    \lambda_i u_i = p_{21}x_i + p_{22}y_i + p_{23}z_i + p_{24}\\
    \lambda_i = p_{31}x_i + p_{32}y_i + p_{33}z_i + p_{34}\\
  \end{align}
$$

Dosazením $\lambda_i$ do rovnice 2 a 3 získáme v maticovém zápisu:
$$
  \begin{align}
        \begin{bmatrix}
            x_i & y_i & z_i & 1 & 0 & 0 & 0 & 0 & -x_i u_i & -y_i u_i & -z_i u_i & -u_i \\
            0 & 0 & 0 & 0 & x_i & y_i & z_i & 1 & -x_i v_i & -y_i v_i & -z_i v_i & -v_i
        \end{bmatrix}
        \begin{bmatrix}
        p_{11} \\
        p_{12} \\
        \vdots \\
        p_{34} \\
        \end{bmatrix}
        = \bf{0}
  \end{align}
$$

Analogicky rozšiřujeme matici korespondencí přidáváním dvojic řádků. Pro $N$ korespondencí skončíme tedy s úlohou:
$$
  \begin{align}
        A^{2N \times 12} \times \bf{p} = \bf{0}
  \end{align}
$$

Tuto rovnici nelze řešit ednoduše vy smyslu nejmenších čtverců, jelikož 
"optimální" řešení je triviální $\bf{p} = \bf{0}$. Proto místo toho řešíme problém pomocí SVD rozkladu:
$$
  \begin{align}
        \min_{||p|| = 1} {||A\bf{p}||^2}
  \end{align}
$$

Optimálním řešením se v tuto chvíli stává pravý vektor z matice $\Sigma$ ze singulárního rozkladu odpovídající nejmenší singulární hodnotě.

Normalizace dat na centroidy s průměrnou vzdáleností $\sqrt{2}$ nebo $\sqrt{3}$ je provedeno pro zvýšení numerické stability algoritmu. Práce s vektory v rozsahu např. 0-1m a 0-3000px by totiž tvořila velké floating-point chyby vevýpočtech.

Následná dekompozice projekční matice $P$ je postavena na RQ-faktorizaci (obdoba QR v opačném pořadí):
$$
  \begin{align}
        P = K [\begin{matrix}
            R_{CB} & t_{CB}
        \end{matrix}] = [\begin{matrix}
            KR_{CB} & Kt_{CB}
        \end{matrix}] = [\begin{matrix}
            M & p_4
        \end{matrix}]
  \end{align}
$$

RQ rozkladem matice M získáme:
$$
  \begin{align}
        M = R Q = KR_{CB} 
  \end{align}
$$
kde $R=K$ je horní trojúhelníková a $Q = R_{CB}$ je ortogonální, přesně jak bychom vyžadovali. Zpětně pak nalezneme translaci:
$$
  \begin{align}
        p_4 = Kt_{CB}\\
        t_{CB} = K^{-1}p_4
  \end{align}
$$

#### Dodatek: PnP metoda

TODO



## Structure of the repository

- `code`: Directory for the assignment implementation.
- `report`: Directory for writing the report and storing any external text sources/figures.

- `sources/bases`: Base plates for holding the different puzzle pieces. It has ArUco 4x4 markers in a diagonal pattern.
- `sources/drawings`: Technical (amateur) drawings of all the parts. **All dimensions are in Milimeters (mm).**
- `sources/print_files`: G-codes for printing all the parts on a Prusa MK3 with a 0.4 nozzle and 3mf files (usual slicer format).
- `sources/robot_ee`: 3D models of adapters for all 3 robot models used (CRS, Bosch and Mitsubishi) and an universal "Hoop" that connects via two magnets onto them.
- `sources/puzzles`: 3D models of different geometric challenges (A,B,C,D,E), again they held to the base plate via two to three magnets.

You'll find every 3D model file in STEP form, which can be opened and edited in any CAD software.

## Hardware information

- Printed out of PETG, 3 perimeters, 10-20% infill. (Cubic or Gyroid infill recommended)
- Robot adapters are two-piece and are designed for easy assembly and disassembly, magnets can be easily harvested, no glue is used.
- Hoop piece is the only part with printed-in magnets.
- Magnets for Hoop: Diameter = 10 mm, height = 5 mm, neodymium.
- Magnets for base & puzzles: Diameter = 10 mm, height = 3 mm, neodymium.
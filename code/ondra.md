## Architektura výpočtu kinematiky (Maze Prediction)

Hlavní myšlenkou modulu je sestavení kompletního kinematického řetězce od základny robota až po cílovou pozici jeho příruby (flange). Inverzní kinematice (IK) se následně předává právě vypočtená transformační matice příruby.

Celková transformace se skládá z tohoto řetězce:
`T_flange = T_base_to_center * T_rotation * T_radius * T_maze_to_hoop * T_hoop_to_flange`

### Jak se počítají jednotlivé transformace

1. **Robot Base -> Rotation Table Center** (`T_base_to_center`)
   - Z nasnímaných dat z ArUco značek si pomocí metody nejmenších čtverců nafitujeme 3 kružnice pro redundanci a vyšší přesnost (trajektorie značky 1, značky 2 a jejich průměru, což reprezentuje základnu bludiště). 
   - Tyto kružnice mají společný střed, čímž získáme absolutní střed rotace celého stolku.

2. **Rotation Table Center -> Angle of Rotation** (`T_rotation`)
   - Na začátku ukotvíme referenční úhel natočení stolku.
   - Aktuální natočení v čase dynamicky predikujeme na základě uplynulého času a odhadnuté úhlové rychlosti ($\omega$).

3. **Angle of Rotation -> Maze Base** (`T_radius`)
   - Slouží k vycentrování přímo na drát bludiště. Jde o translační posun osy odpovídající naměřenému poloměru kružnice středového bodu.

4. **Maze Base -> Hoop Position** (`T_maze_to_hoop`)
   - Vypočteno pomocí parametrizace dráhy konkrétního bludiště (např. v modulu `maze_d.py`).
   - Určuje exaktní translační pozici a základní orientaci obruče pro projetí daného bodu na křivce.

5. **Hoop Position -> Flange Position** (`T_hoop_to_flange`)
   - Aplikace offsetu nástroje. Určíme si lokální transformaci `T_flange_to_hoop` (fyzické uchycení držáku na rameni) a do rovnice vložíme její inverzi. Tím získáme exaktní pozici a orientaci pro holou přírubu.

---

### Známé problémy a prostor pro optimalizaci (To-Do)
* **Singularity a protáčení kloubů:** Při velmi malé změně cílové pozice/natočení občas IK vybere řešení, které si vynutí protočení kloubu (typicky zápěstí) téměř o 360°, což způsobuje neplynulý pohyb.
* **Přeurčená orientace obruče:** Parametrizace (např. v `maze_d.py`) striktně definuje celou 3D orientaci obruče v prostoru. Pro úspěšné projetí bludištěm však reálně stačí, aby byla rovina obruče kolmá k drátu – rotace obruče *okolo* samotné osy drátu může být libovolná. Uvolnění tohoto stupně volnosti by algoritmu umožnilo nacházet plynulejší trajektorie s menším pohybem kloubů a lépe předcházet kolizím ramene s podložkou.
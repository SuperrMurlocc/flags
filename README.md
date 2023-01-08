<p><strong>Streszczenie</strong>: Sprawozdanie opisuje projekt, w którym
sieć neuronowa została użyta do klasyfikowania obrazów flag
europejskich. Sieć została wytrenowana na zbiorze danych składającym się
z obrazów flag należących do 27 różnych krajów. Autorzy zaimplementowali
również techniki zapobiegające przeuczeniu, takie jak augmentacja danych
i wczesne zatrzymywanie. Sprawozdanie wnosi, że sieć neuronowa była
skuteczna w klasyfikowaniu obrazów flag europejskich z dużą
dokładnością, również dla zbioru testowego, który zawierał obrazy
niewidzianie przez sieć w procesie uczenia.</p>
<p><strong>Słowa kluczowe</strong>: Sieć neuronowa, klasyfikacja
obrazów, flagi</p>
<h1 id="wstęp">Wstęp</h1>
<p>Celem projektu było stworzenie programu klasyfikcującego obrazy
flag.</p>
<h2 id="założenia-projektowe">Założenia projektowe</h2>
<p>Analizowany obraz i flaga na nim wybierane są przez użytkownika w
interfejsie graficznym. Sieć neuronowa dokonuje predyckji na podstawie
zaznaczonego fragmentu obrazu i przedstawia maksymalnie trzy najbardziej
podobne flagi, których skala podobieństwa przekroczyła 20%.</p>
<h2 id="podobne-rozwiązania">Podobne rozwiązania</h2>
<p>Nie znaleziono żadnego podobnego rozwiązania istniejącego już na
rynku. Najbardziej zbliżona do naszego rozwiązania jest strona <a
href="https://flagid.org/">FLAGID</a><span class="citation"
data-cites="flagid">[1]</span>, która pozwala identyfikować flagi, ale
jedynie na podstawie filtrów wybranych przez użytkownika, tj. kolorów
występująych na fladze, jej kształtu itp.</p>
<h1 id="materiały-i-metody">Materiały i metody</h1>
<h2 id="dane">Dane</h2>
<p>Do nauczenia sieci neuronowej zebrano 1109 obrazów flag należących do
27 klas, gdzie każda klasa była innym państwem. Posłużono się gotowym
zestawem obrazów ze strony <a
href="https://www.kaggle.com/datasets/yusufyldz/countries-flags-images">Kaggle</a><span
class="citation" data-cites="kaggle:flagImages">[2]</span>, do którego
dołożono kilka klas samodzielnie znajdując obrazy przy pomocy
wyszukiwarki <a href="https://www.google.com/imghp">Google</a><span
class="citation" data-cites="google_images">[3]</span>. Obrazy wybierano
tak aby przedstawiały one flagi w najróżniejszy sposób, uwzględniając
otoczenie, oświetlenie i ułożenie flagi przy zdjęciach, oraz kształt
przy grafikach.</p>
<p>W tablicy <a href="#tbl:austria_examples">1</a> można
zauważyć różnorodność dobieranych obrazów na przykładzie dziewięciu
obrazów klasy <code>Austria</code>.</p>
<div id="tbl:austria_examples" class="tablenos">
<table id="tbl:austria_examples">
<caption><span>Tablica 1:</span> Przykładowe obrazy flag, należące do
klasy <code>Austria</code>. </caption>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/1.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/2.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/3.jpg"
title="fig:" height="66" /></td>
</tr>
<tr class="even">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/4.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/5.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/6.png"
title="fig:" height="66" /></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/7.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/8.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria/9.jpg"
title="fig:" height="66" /></td>
</tr>
</tbody>
</table>
</div>
<p>Wszystkie obrazy przy wczytywaniu były również przeskalowywane do
rozmiarów <code>300x200</code> aby osiągnąć najpopularniejsze proporcje
flagi, czyli <code>3:2</code>.</p>
<h2 id="metody">Metody</h2>
<h3 id="sieć-neuronowa">Sieć neuronowa</h3>
<p>Wykorzystano model <code>Sequential</code> z biblioteki
<code>tensorflow.keras</code>. Jest to liniowy stos warstw, w którym
można korzystać z dużej różnorodności warstw dostepnych w bibliotece
Keras<span class="citation"
data-cites="tensorflow:sequential">[4]</span>. W tablicy <a
href="#tbl:model_layers">2</a> zobaczyć można wykorzystane warstwy,
rozmiary przyjmowanych wejść, oraz liczbę wag wejściowych.</p>
<div id="tbl:model_layers" class="tablenos">
<table id="tbl:model_layers">
<caption><span>Tablica 2:</span> Warstwy modelu sieci neuronowej.
</caption>
<colgroup>
<col style="width: 47%" />
<col style="width: 38%" />
<col style="width: 13%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">Rodzaj warstwy</th>
<th style="text-align: left;">Rozmiar wejść</th>
<th style="text-align: left;">Liczba wag wejściowych</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Rescaling</td>
<td style="text-align: left;">(?, 200, 300, 3)</td>
<td style="text-align: left;">0</td>
</tr>
<tr class="even">
<td style="text-align: left;">Conv2D</td>
<td style="text-align: left;">(?, 200, 300, 16)</td>
<td style="text-align: left;">448</td>
</tr>
<tr class="odd">
<td style="text-align: left;">MaxPooling2D</td>
<td style="text-align: left;">(?, 100, 150, 16)</td>
<td style="text-align: left;">0</td>
</tr>
<tr class="even">
<td style="text-align: left;">Conv2D</td>
<td style="text-align: left;">(?, 100, 150, 32)</td>
<td style="text-align: left;">4640</td>
</tr>
<tr class="odd">
<td style="text-align: left;">MaxPooling2D</td>
<td style="text-align: left;">(?, 50, 75, 32)</td>
<td style="text-align: left;">0</td>
</tr>
<tr class="even">
<td style="text-align: left;">Conv2D</td>
<td style="text-align: left;">(?, 50, 75, 64)</td>
<td style="text-align: left;">18496</td>
</tr>
<tr class="odd">
<td style="text-align: left;">MaxPooling2D</td>
<td style="text-align: left;">(?, 25, 37, 64)</td>
<td style="text-align: left;">0</td>
</tr>
<tr class="even">
<td style="text-align: left;">Flatten</td>
<td style="text-align: left;">(?, 59200)</td>
<td style="text-align: left;">0</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Dense</td>
<td style="text-align: left;">(?, 128)</td>
<td style="text-align: left;">7577728</td>
</tr>
<tr class="even">
<td style="text-align: left;">Dense</td>
<td style="text-align: left;">(?, 27)</td>
<td style="text-align: left;">3483</td>
</tr>
</tbody>
</table>
</div>
<p>Do sieci neurnowej przekazujemy wejście o rozmiarze (?, 200, 300, 3).
Znak zapytania <code>?</code> oznacza zmienną liczbę wejść, będącą
liczbą przekazanych obrazów. Trzy kolejne rozmiary 200, 300 i 3
oznaczają wymiary obrazu - jego wysokość, szerokść oraz
liczbę składowych w przestrzeni barw obrazu.</p>
<p>Wyjście ma rozmiar (?, 27). Pierwszy rozmiar tak jak poprzednio jest
liczbą przekazanych obrazów, natomiast drugi jest liczbą klas. Oznacza
to że na wyjściu mamy tablicę tablic, gdzie każda wewnętrzna tablica
składa się z 27 liczb, każda oznaczająca skalę podobieństwa do danej
klasy.</p>
<p>Model składa się z 10 warstw:</p>
<ul>
<li>Rescaling - normalizująca obraz, zamieniając wartości pikseli z
przedziału <code>0 - 255</code> do <code>0 - 1</code>;</li>
<li>trzy pary warstw:
<ul>
<li>Conv2D - dokunująca operacji konwolucyjnych na obrazie w celu
wydobycia cech opisujących obraz;</li>
<li>MaxPooling2D - ograniczająca liczbę cech wydobytych w poprzedniej
warstwie;</li>
</ul></li>
<li>Flatten - przekonwertywująca wielo-wymiarową tablicę do takiej o
pojedynczym rozmiarze, aby móc przekazać ją do warstwy Dense;</li>
<li>dwie warstwy Dense doknujące predykcji na podstaw wag
wejściowych.</li>
</ul>
<h3 id="prewencja-przed-przeuczeniem">Prewencja przed przeuczeniem</h3>
<p>Wyżej opisany model sieci neuronowej szybko się przeuczał. Aby temu
zapobiec postanowiono dodać dwie warstwy do sieci neuronowej.</p>
<p>Przed wszystkimi warstwami dodano warstwę augmentacji danych.
Tensorflow umożliwia automatyczną danych poprzez parametry takie jak
losowe przybliżenie, losowe obrócenie oraz losowe lustrzane odbicie w
pionie lub poziomie<span class="citation"
data-cites="tensorflow:dataAugmentation">[5]</span>. Wykorzystano tylko
dwa pierwsze, ponieważ lustrzane odbicie flagi może być flagą innego
państwa, jak przykładowo flaga Monako i Polski, przedstawione w tablicy
<a href="#tbl:poland_and_monako">3</a> poniżej.</p>
<div id="tbl:poland_and_monako" class="tablenos">
<table id="tbl:poland_and_monako">
<caption><span>Tablica 3:</span> Dwie flagi będące swoim lustrzanym
odbiciem. </caption>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/poland_monako/monako.jpg"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/poland_monako/poland.png"
title="fig:" height="66" /></td>
</tr>
</tbody>
</table>
</div>
<p>Na rysunku <a href="#fig:estonia_augmentation">1</a> przedstawiono
przykładowy efekt augmentacji na jednym z obrazów flagi Estonii.</p>
<div id="fig:estonia_augmentation" class="fignos">
<figure>
<img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/data_augmentation/augmentation_effect.png"
alt="Figure 1: Efekt augmentacji" />
<figcaption aria-hidden="true"><span>Rysunek 1:</span> Efekt
augmentacji</figcaption>
</figure>
</div>
<p>Dodatkowo przed warstwą <em>Flatten</em> dodano
warstwę <em>Dropout</em>. Warstwa ta losowo zeruje ustaloną ilość wag
wejściowych. W ten sposób sieć nie może za bardzo polegać na tych samych
wagach, zapobiegając tym samym przeuczenie się. Zadawalające efekty
osiągnięto po ustaleniu ilości zerowanych wag wejściowych na 20%.</p>
<p>Taki model sieci neuronowej pozwolił na uzyskanie zadawaląjących
wyników.</p>
<h3 id="wycinek-obrazu">Wycinek obrazu</h3>
<p>Wycinek obrazu tworzony był poprzez zmianę perspektywy oryginalnego
obrazu. Transformacja perspektywy była definiowana przez macierz
przekształcenia liniowego. Macierz pozyskiwana była przy pomocy funkcji
<code>getPerspectiveTransform</code>, a następnie aplikowana na obraz w
celu wydobycia wycinka przy pomocy funkcji <code>warpPerspective</code>
biblioteki <a
href="https://docs.opencv.org/4.x/index.html">openCV</a><span
class="citation" data-cites="opencv">[6]</span>.</p>
<h2 id="technologie-i-narzędzia">Technologie i narzędzia</h2>
<p>Do stworzenia modelu użyto bibliotek:</p>
<ul>
<li><code>Tensorflow</code></li>
<li><code>Keras</code></li>
</ul>
<p>W szczególności posłużono się poradnikiem Tensorfow dotyczącym
klasyfikacji obrazów<span class="citation"
data-cites="tensorflow:imageClassification">[7]</span>.</p>
<p>Do przetwarzania obrazów wykorzystno bibliotekę
<code>openCV</code>.</p>
<p>Dodatkowo zespół wykorzystywał GitHuba, jako zdalne repozytorium
kodu. Repozytorium dostępne jest pod tym adresem: <a
href="https://github.com/SuperrMurlocc/flags"><strong>github.com/SuperrMurlocc/flags</strong></a>.</p>
<h1 id="wyniki">Wyniki</h1>
<h2 id="jakość-klasyfikacji">Jakość klasyfikacji</h2>
<h3 id="dokładność">Dokładność</h3>
<p>W tablicy <a href="#tbl:accuracyTable">4</a> przedstawiono wykresy
zależności dokładności klasyfikacji dla zbioru uczącego (<em>Training
Accuracy</em>) i testowego (<em>Validation Accuracy</em>) od epoki
uczenia. Po lewej widać wykres dla sieci przeuczonej a po prawej
ostateczne osiągnięte wyniki.</p>
<div id="tbl:accuracyTable" class="tablenos">
<table id="tbl:accuracyTable">
<caption><span>Tablica 4:</span> Zależność dokładności od epoki uczenia.
</caption>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/accuracy/overfitted.png"
title="fig:" height="500" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/accuracy/fitted.png"
title="fig:" height="500" /></td>
</tr>
</tbody>
</table>
</div>
<p>Na lewym wykresie wyraźnie widać przeuczenie się sieci neuronowej.
Podczas gdy zbiór treningowy kontynuuje osiągać coraz to
lepszą dokładność, to zbiór testowy zatrzymuje się na poziomie około 70%
dokładności i od 3 epoki drastycznie spada wzrost dokładności. Na prawym
wykresie efekt przeuczenia jest znacznie mniejszy, a
sieć osiąga dokładność ponad 80% na zbiorze testowym.</p>
<h3 id="macierz-konfuzji">Macierz konfuzji</h3>
<p>W celu dokładniejszego zbadania jakości klasyfikacji sieci neuronowej
stworzono macierz konfuzji. Dane zbierano pięćdziesięciokrotnie biorąc
20% losowych obrazów i porównując faktyczną klasę do predykcji sieci
neuronowej. Wyniki przedstawiono na rysunku <a
href="#fig:confusionMatrix">2</a>.</p>
<div id="fig:confusionMatrix" class="fignos">
<figure>
<img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/confusion_matrix/confusionMatrix.png"
alt="Figure 2: Macierz konfuzji" />
<figcaption aria-hidden="true"><span>Rysunek 2:</span> Macierz
konfuzji</figcaption>
</figure>
</div>
<p>Macierz przedstawia zależność faktycznej klasy od tej predykowanej
przez model. Wyniki na głównej przekątnej to te poprawne - kiedy
predykowana (<em>predicted</em>) klasa jest tą rzeczywistą
(<em>actual</em>). Wszystkie inne predykcje są błędne. Dla danej klasy
możemy określić trzy typy wyników:</p>
<ul>
<li>True Positive (<em>TP</em>) - wtedy, kiedy klasa rzeczywista zgadza
się z klasą predykowaną;</li>
<li>False Positive (<em>FP</em>) - wtedy, kiedy to ta klasa była klasą
predykowaną, a rzeczywista była inna;</li>
<li>False Negative (<em>FN</em>) - wtedy, kiedy to ta klasa była klasą
rzeczywistą, a predykowana była inna.</li>
</ul>
<p>Na podstawie tych trzech typów wyników, dla każdej klasy
wyznaczono trzy wskaźniki opisujące jakość modelu.</p>
<ol type="1">
<li>Czułość (<em>recall</em>), czyli wartość opisująca zdolność sieci
neuronowej do wykrycia danej klasy. Czułość opisywana jest wzorem <span
class="math inline">$\frac{\Sigma{} TP}{\Sigma{} TP+\Sigma{}
FN}$</span>;</li>
<li>Precyzja (<em>precision</em>), czyli stopień zgodności między
uzyskanymi wynikami, określana wzorem <span
class="math inline">$\frac{\Sigma{} TP}{\Sigma{} TP+\Sigma{}
FP}$</span>;</li>
<li>F1, czyli miara balansu pomiędzy czułością i precyzją. Określana
jest wzorem <span
class="math inline">$\frac{2\Sigma{}TP}{2\Sigma{}TP+\Sigma{}FP+\Sigma{}FN}$</span>.</li>
</ol>
<h1 id="dyskusja-i-wnioski">Dyskusja i wnioski</h1>
<p>Na podstawie trzech wskaźników macierzy konfuzji możemy zauważyć, że
najgorzej klasyfikowane są flagi Austrii, Danii oraz Łotwy, zapewne ze
względu na bardzo podobne flagi, które można zobaczyć w tablicy <a
href="#tbl:austria_denmark_latvia">5</a> poniżej.</p>
<div id="tbl:austria_denmark_latvia" class="tablenos">
<table id="tbl:austria_denmark_latvia">
<caption><span>Tablica 5:</span> Flagi Austrii, Danii oraz Łotwy.
</caption>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria_denmark_latvia/austria.png"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria_denmark_latvia/denmark.png"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria_denmark_latvia/latvia.png"
title="fig:" height="66" /></td>
</tr>
</tbody>
</table>
</div>
<p>Najlepiej klasyfikowane natomiast są flagi państw: Bułgarii, Czech
oraz Niemiec. Zapewne ze względu na brak flag o podobnych kolorach
(Bułgaria i Niemcy) lub kształtach (Czechy).</p>
<div id="tbl:other_countries" class="tablenos">
<table id="tbl:other_countries">
<caption><span>Tablica 6:</span> Flagi Bułgarii, Czech i Niemiec.
</caption>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria_denmark_latvia/bulgaria.png"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria_denmark_latvia/czech.png"
title="fig:" height="66" /></td>
<td style="text-align: center;"><img
src="https://github.com/SuperrMurlocc/flags/blob/master/imgs/austria_denmark_latvia/germany.png"
title="fig:" height="66" /></td>
</tr>
</tbody>
</table>
</div>
<p>Jednak dla wszystkich państw wszystkie wskaźniki osiągnęły
zadawaląjące(&gt; 80%) wyniki. Uczenie maszynowe jest narzędziem dzięki
któremu klasyfikacji obrazów można dokonać łatwiej i szybciej.</p>
<p>Należy jednak mieć na uwadze, że taki model poprawnie rozpozna tylko
te flagi, których jest nauczony. Próbując przedstawić mu flagę państwa,
którego nie został nauczony, nie otrzymamy dobrego wyniku. Jest to jedna
z głównych wad uczenia maszynowego, będąca szczególnie istotna przy
systemach mających wpływ na zdrowie i życie człowieka. Przykładowo,
uczenie maszynowe może nie być najlepszym narzędziem do tworzenia
zuatomatyzowanych samochodów, ponieważ w sytuacji krytycznej, takiej,
jakiej nie został on wcześniej nauczony, otrzymamy niezdefiniowane
zachowanie.</p>
<h1 class="unnumbered" id="bibliography">Bibliografia</h1>
<div id="refs" class="references csl-bib-body" role="doc-bibliography">
<div id="ref-flagid" class="csl-entry" role="doc-biblioentry">
<div class="csl-left-margin">[1] </div><div
class="csl-right-inline"><span>„Identify a flag”</span>, <em>Identify a
Flag - Flag identifier</em>. Dostępne na: <a
href="https://flagid.org/">https://flagid.org/</a></div>
</div>
<div id="ref-kaggle:flagImages" class="csl-entry"
role="doc-biblioentry">
<div class="csl-left-margin">[2] </div><div class="csl-right-inline">S.
Aylık, <span>„Countries flags images”</span>, <em>Kaggle</em>. lipiec
2022. Dostępne na: <a
href="https://www.kaggle.com/datasets/yusufyldz/countries-flags-images">https://www.kaggle.com/datasets/yusufyldz/countries-flags-images</a></div>
</div>
<div id="ref-google_images" class="csl-entry" role="doc-biblioentry">
<div class="csl-left-margin">[3] </div><div
class="csl-right-inline"><em>Google Images</em>. Google. Dostępne na: <a
href="https://www.google.com/imghp">https://www.google.com/imghp</a></div>
</div>
<div id="ref-tensorflow:sequential" class="csl-entry"
role="doc-biblioentry">
<div class="csl-left-margin">[4] </div><div
class="csl-right-inline"><span>„Tf.keras.sequential: tensorflow
V2.11.0”</span>, <em>TensorFlow</em>. Dostępne na: <a
href="https://www.tensorflow.org/api_docs/python/tf/keras/Sequential">https://www.tensorflow.org/api_docs/python/tf/keras/Sequential</a></div>
</div>
<div id="ref-tensorflow:dataAugmentation" class="csl-entry"
role="doc-biblioentry">
<div class="csl-left-margin">[5] </div><div
class="csl-right-inline"><span>„Data augmentation: Tensorflow
Core”</span>, <em>TensorFlow</em>. Dostępne na: <a
href="https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en">https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en</a></div>
</div>
<div id="ref-opencv" class="csl-entry" role="doc-biblioentry">
<div class="csl-left-margin">[6] </div><div
class="csl-right-inline"><span>„OpenCV modules”</span>, <em>OpenCV</em>.
Dostępne na: <a
href="https://docs.opencv.org/4.x/index.html">https://docs.opencv.org/4.x/index.html</a></div>
</div>
<div id="ref-tensorflow:imageClassification" class="csl-entry"
role="doc-biblioentry">
<div class="csl-left-margin">[7] </div><div
class="csl-right-inline"><span>„Image classification: Tensorflow
Core”</span>, <em>TensorFlow</em>. Dostępne na: <a
href="https://www.tensorflow.org/tutorials/images/classification?hl=en">https://www.tensorflow.org/tutorials/images/classification?hl=en</a></div>
</div>
</div>

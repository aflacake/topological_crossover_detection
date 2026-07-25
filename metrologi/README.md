# Topological Involution Constant

## Hipotesis Awal

Jika suatu antarmuka masih bersifat fraktal dengan dimensi

$$d_f^{\text{crit}} = D - \frac{\beta}{\nu} = 3 - \frac{0.418}{0.876} \approx 2.52$$

maka elastisitas alometriknya diperkirakan mengikuti

$$K_{\text{trans}} = \frac{d_f^{\text{crit}}}{3} = \frac{2.52}{3} = 0.84$$

Hipotesis ini secara implisit mengasumsikan bahwa setiap pertambahan volume tetap menghasilkan antarmuka baru secara proporsional sebagaimana pada rezim fraktal biasa.

> Dengan kata lain, tidak ada mekanisme internal yang secara sistematis menghilangkan luas permukaan yang telah terbentuk.

## Konflik dengan Simulasi

Namun simulasi pertama justru menunjukkan sesuatu yang tidak sesuai.

Ketika sistem mencapai

$\chi = 0$

eksponen yang terukur bukan

$0.84$

melainkan

$0.47$

Perbedaan sebesar ini terlalu besar untuk dianggap sebagai kesalahan numerik.

> Artinya, asumsi dasar model fraktal murni ternyata tidak lagi berlaku setelah sistem memasuki rezim topologi yang sangat terhubung.

## Mencari Penyebab

Analisis konfigurasi kisi memperlihatkan adanya mekanisme baru yang tidak ada pada model fraktal sederhana.

Ketika volume bertambah, tidak semua voxel baru menciptakan permukaan baru.

Sebaliknya, sebagian voxel justru menjembatani dua cabang yang sebelumnya terpisah.

Akibatnya terbentuk loop baru dan antarmuka internal menghilang.

Secara konseptual,

$$K_{\text{trans}} = \frac{V}{A} \left( \underbrace{\frac{\partial A}{\partial V}_{\text{murni}}}_{\text{Positif (+) Pertambahan Luas}} + \underbrace{\frac{\partial A}{\partial b_1} \cdot \frac{db_1}{dV}}_{\text{Negatif (-) Kanibalisasi Loop}} \right)$$

Jadi pertumbuhan luas permukaan bukan lagi proses satu arah.

Ia menjadi hasil kompetisi antara dua mekanisme.

## Menyadari Bahwa Ada Dua Titik Kritis

Menyadari bahwa selama ini dua kondisi berbeda telah tercampur.

### Ambang Perkolasi

$p_c$

Menandai munculnya cluster raksasa pertama.

Dominannya masih pertumbuhan massa.

Loop relatif sedikit.

### Kelintasan Nol Euler

$\chi = 0$

Menandai keadaan ketika jumlah loop hampir menyeimbangi jumlah komponen.

Di sini topologi telah matang.

Rekoneksi antarmuka menjadi mekanisme dominan.

Kedua titik tersebut tidak ekuivalen.

Karena itu tidak ada alasan teoritis mengapa eksponen skalanya harus sama.

## Uji Finite-Size Scaling

Memastikan bahwa hasil tersebut bukan artefak ukuran kisi, dilakukan Finite-Size Scaling.

### Percobaan A

Didapat
$\approx 0.5115$

Sekilas hasil ini terlihat menarik karena mendekati
$\frac{1}{2}$

Namun kualitas statistiknya lemah,
$0.4370$

Artinya estimasi sangat sensitif terhadap fluktuasi realisasi dan efek batas.

Eksponen ini tidak menunjukkan konvergensi yang kuat.

### Percobaan B

Sebaliknya, nilai yang diperoleh berubah secara sistematis

$L = 32$ : $0.5331 \pm 0.0413$

$L = 64$ : $0.4660 \pm 0.0171$

$L = 64$ : $0.4660 \pm 0.0171$

dan setelah diekstrapolasi

$L \to \infty$

memberikan

$\approx 0.3819$

Yang jauh lebih penting,

$0.9881$

serta simpangan baku turun hingga

$0.0013$

Ini menunjukkan adanya konvergensi yang kuat menuju satu nilai limit.

Dengan demikian, eksponen

$\approx 0.3819$

bukan sekadar angka hasil fitting.

Ia merupakan nilai tetap yang muncul ketika dua mekanisme berikut mencapai keseimbangan statistik.

Dalam limit kontinu,

kedua mekanisme tersebut menghasilkan hukum skala

$$A(V) \propto V^{0.3819}$$

karena setiap penambahan volume sekaligus mempercepat rekoneksi internal yang menghilangkan sebagian permukaan.
![Wolfram spirals](https://github.com/Ionosi/assets/blob/main/shadows.jpg?raw=true)
# Visualization of Polynomial Shadows

This repository contains Mathematica code for plotting approximations of the shadows of polynomials. These approximations consist of the union of all roots of all orders of the derivative of a polynomial raised to a specified power. The asymptotic objects, emerging as the power approaches infinity, are discussed in my article "Rodrigues' Descendants of a Polynomial and Boutroux Curves" at https://doi.org/10.1007/s00365-023-09657-x

## Usage

To use the code, you need Mathematica installed on your computer. Open the .nb file in Mathematica, and run the script. The main parameters you can adjust are:

- **`P`**: The polynomial to be analyzed.
- **`n`**: The exponent to which the polynomial is raised.

## Example Plot

The default configuration uses the polynomial **`P=z(z-12)(z-2-8i)(z-8-7i)`** and **`n=15`**. The script will generate a complex plot showing:

- The roots of **`P`** (black points).
- The roots of all derivatives of **`P^n`** (red points).
- Branch points of an algebraic equation for the Cauchy transform (blue rectangles).
- Roots of **`P'`** (green rectangles).
- The center of mass of the zeros of **`P`** (green triangle).

## Contributing

Contributions to this project are welcome. You can contribute in various ways:

- By suggesting improvements to the code.
- By extending the code for rational functions.
- By reporting bugs or issues.

## License

This project is licensed under the MIT License.

## Acknowledgments

This work is based on the findings presented in the article "Rodrigues' Descendants of a Polynomial and Boutroux Curves", which I wrote with Rikard Bögvad and Boris Shapiro.

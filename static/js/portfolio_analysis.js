document
  .getElementById("analyze-portfolio")
  .addEventListener("click", async () => {
    const tickers = document
      .getElementById("tickers")
      .value.split(",")
      .map((t) => t.trim());
    const weights = document
      .getElementById("weights")
      .value.split(",")
      .map((w) => parseFloat(w.trim()));
    const currency = document.getElementById("currency").value;

    if (tickers.length !== weights.length) {
      alert("The number of tickers and weights must match.");
      return;
    }

    try {
      // Send portfolio data to the backend
      const response = await fetch("/portfolio_analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers, weights, currency }),
      });

      const result = await response.json();
      if (response.ok) {
        // Display metrics
        document.getElementById("expected-return").textContent =
          result.metrics.expected_return.toFixed(2);
        document.getElementById("risk").textContent =
          result.metrics.risk.toFixed(2);
        document.getElementById("sharpe-ratio").textContent =
          result.metrics.sharpe_ratio.toFixed(2);
        document.getElementById("diversification-ratio").textContent =
          result.metrics.diversification_ratio.toFixed(2);
        document.getElementById("historical-cagr").textContent =
          result.metrics.historical_cagr.toFixed(2);

        // Show the results section
        document.getElementById("portfolio-results").style.display = "block";

        // Render efficient frontier chart
        const ctx = document
          .getElementById("efficientFrontierChart")
          .getContext("2d");
        const frontierData = result.efficient_frontier; // Efficient frontier data from backend
        new Chart(ctx, {
          type: "scatter",
          data: {
            datasets: [
              {
                label: "Efficient Frontier",
                data: Object.keys(frontierData).map((key) => ({
                  x: parseFloat(key), // Risk
                  y: parseFloat(frontierData[key]), // Return
                })),
                backgroundColor: "rgba(54, 162, 235, 0.2)",
                borderColor: "rgba(54, 162, 235, 1)",
                pointRadius: 5,
              },
            ],
          },
          options: {
            scales: {
              x: { title: { display: true, text: "Risk (%)" } },
              y: { title: { display: true, text: "Return (%)" } },
            },
          },
        });
      } else {
        alert(
          result.error || "An error occurred while analyzing the portfolio."
        );
      }
    } catch (error) {
      alert("Failed to fetch portfolio analysis results. Please try again.");
      console.error(error);
    }
  });

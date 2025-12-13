document.getElementById("surveyForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const data = {
    Age_Group: document.getElementById("age").value,
    Gender: document.getElementById("gender").value,
    Sleep_Hours: document.getElementById("sleep").value,
    Screen_Time: document.getElementById("screen").value,
    Free_Time_Activity: document.getElementById("free").value,
    Exercise_Frequency: document.getElementById("exercise").value,
    Anxious_Exhausted_Frequency: document.getElementById("anxiety").value,
    Current_Emotional_State: document.getElementById("emotion").value
  };

  const response = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const result = await response.json();

  document.getElementById("result").innerHTML = `
    <h3>Prediction Result</h3>
    <p><b>Stress Risk:</b> ${result.stress_risk}</p>
    <p><b>Help Recommended:</b> ${result.help_recommended}</p>
    <p><b>Risk Cluster:</b> ${result.risk_cluster}</p>
  `;
});

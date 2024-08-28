function validateForm() {
    let isValid = true;
    const fields = ['area', 'bhk', 'bathroom', 'isFurnished', 'parking', 'isApartment'];

    fields.forEach(field => {
      const input = document.getElementById(field);
      const errorDiv = document.getElementById(`error-${field}`);

      if (input.value.trim() === '' || (input.tagName === 'SELECT' && input.value === '')) {
        isValid = false;
        errorDiv.style.display = 'block';
      } else {
        errorDiv.style.display = 'none';
      }
    });

    return isValid;
  }

  function submitForm() {
    if (validateForm()) {
      const form = document.getElementById('dataForm');
      fetch('http://localhost:5000/submit-form', {
        method: 'POST',
        body: new FormData(form),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        document.getElementById('response').innerHTML = `
          <div class="alert alert-success" role="alert">
            Price: ₹${Math.round(data.price)}<br/>
            *This is the estimated Price. The Original Price may vary and it depends on many factors such as popularity of area, design, infrastructure, place and many more.
          </div>
        `;
      })
      .catch(error => {
        document.getElementById('response').innerHTML = `
          <div class="alert alert-danger" role="alert">
            Error: ${error.message}
          </div>
        `;
      });
    }
  }
$(document).ready(function () {
    $('.custom-file-input').on('change', function() {
        let fileName = $(this).val().split('\\').pop(); 
        $(this).next('.custom-file-label').addClass("selected").html(fileName); 
     });
     
     $('.custom-file-input').click(function() {
      $('.error-feedback').hide();
    });


    $('#fileSubmitForm').submit(function (event) {
        event.preventDefault();
        event.stopPropagation();
        var formData = new FormData();
        var fileSelect = document.getElementById("fileSelect");
        if (fileSelect.files && fileSelect.files.length == 1) {
            var file = fileSelect.files[0];
            formData.set("file", file, file.name);
        }

        $.ajax({
            // Your server script to process the upload
            url: 'http://127.0.0.1:5000/timeforcasting/predict',
            type: 'POST',

            // Form data
            data: formData,

            // Tell jQuery not to process data or worry about content-type
            // You *must* include these options!
            cache: false,
            contentType: false,
            processData: false,

            // Custom XMLHttpRequest
            xhr: function () {
                var myXhr = $.ajaxSettings.xhr();
                $('.lds-grid').fadeIn();
                if (myXhr.upload) {
                    // For handling the progress of the upload
                    myXhr.upload.addEventListener('progress', function (e) {
                        if (e.lengthComputable) {
                            $('progress').attr({
                                value: e.loaded,
                                max: e.total,
                            });
                        }
                    }, false);
                }
                return myXhr;
            },
            success: function (data) {
                $('.lds-grid').hide();
                createAreaChart(data.timestamp, data.inputBPM, data.predictBPM, data.max, data.min);
                $('.dataCard').fadeIn('slow');
            },
            error: function (data) {
                $('.lds-grid').hide();
                $('.error-feedback').fadeIn('slow');
                $('.error-feedback').html("Error: " + data.responseJSON.message);
            }
        });
    });
});


// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Area Chart 
function createAreaChart(timestamp, inputBPM, predictBPM, max, min){
    //var allBPM = (inputBPM.concat(predictBPM));
    var maxYAxis = Math.ceil((Math.max(max) + 5)/5)*5;
    var minYAxis = Math.ceil((Math.min(min) - 5)/5)*5;
    var ctx = document.getElementById("myAreaChart");
    var myLineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: timestamp,
        datasets: [{
          label: "Input (bpm)",
          lineTension: 0.3,
          backgroundColor: "rgba(2,117,216,0.2)",
          borderColor: "rgba(2,117,216,1)",
          pointRadius: 5,
          pointBackgroundColor: "rgba(2,117,216,1)",
          pointBorderColor: "rgba(255,255,255,0.8)",
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "rgba(2,117,216,1)",
          pointHitRadius: 50,
          pointBorderWidth: 2,
          data: inputBPM,
          fill : false
        },
        {
          label: "Predicted (bpm)",
          lineTension: 0.3,
          backgroundColor: "rgba(2,230,130,0.2)",
          borderColor: "rgba(2,230,130,1)",
          pointRadius: 5,
          pointBackgroundColor: "rgba(2,230,130,1)",
          pointBorderColor: "rgba(255,255,255,0.8)",
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "rgba(2,117,216,1)",
          pointHitRadius: 50,
          pointBorderWidth: 2,
          data: predictBPM,
          fill : false
        }],
      },
      options: {
        scales: {
          xAxes: [{
            time: {
              unit: 'time'
            },
            gridLines: {
              display: false
            },
            ticks: {
              maxTicksLimit: 7
            }
          }],
          yAxes: [{
            ticks: {
              min: minYAxis,
              max: maxYAxis,
              maxTicksLimit: 15
            },
            gridLines: {
              color: "rgba(0, 0, 0, .125)",
            }
          }],
        },
        legend: {
          display: true
        }
      }
    });
    
}


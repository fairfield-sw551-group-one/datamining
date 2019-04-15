$(document).ready(function () {

    $('.custom-file-input').on('change', function () {
        let fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').addClass("selected").html(fileName);
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
            url: 'http://127.0.0.1:5000/ecgclassification/predict',
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
                addDataToTable(data)
                addDataToPieChart(data)
                $('.dataCard').fadeIn('slow');
            },
            error: function (data) {
                var json = $.parseJSON(data);
                alert(json.error);
            }
        });
    });
});

function addDataToTable(data) {
    var t = $('#dataTable').DataTable();
    $.each(data, function (i) {
        t.row.add([data[i].beatId,
        data[i].timestamp,
        data[i].type,
        data[i].confidence]).draw(false);
    })
}

function addDataToPieChart(data) {
    var typeArray = data.map(x => x.type);
    var typeCounts = {};
    typeArray.forEach(function(x) { typeCounts[x] = (typeCounts[x] || 0)+1; });
    var ctx = document.getElementById("pieChart");
    var myPieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(typeCounts),
            datasets: [{
                data: Object.values(typeCounts),
                backgroundColor: ['#007bff', '#dc3545', '#ffc107', '#28a745'],
            }],
        },
    });
}
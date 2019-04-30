$(document).ready(function () {

    $('#APCinfo, #PVCinfo, #LBBBinfo, #RBBBinfo, #VEinfo, #Ninfo').hide();

    $('.custom-file-input').on('change', function () {
        let fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').addClass("selected").html(fileName);
    });

    $('.custom-file-input').click(function () {
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
                $('.lds-grid').hide();
                $('.error-feedback').fadeIn('slow');
                $('.error-feedback').html("Error: " + data.responseJSON.message);
            }
        });
    });
});

function addDataToTable(data) {
    var t = $('#dataTable').DataTable();
    $.each(data, function (i) {
        t.row.add([data[i].beatId,
        data[i].start,
        data[i].end,
        data[i].type]).draw(false);
    })
}

function addDataToPieChart(data) {
    var typeArray = data.map(x => x.type);
    var typeCounts = {};
    typeArray.forEach(function (x) { typeCounts[x] = (typeCounts[x] || 0) + 1; });
    addPredictionDescreiptions(typeCounts);
    var ctx = document.getElementById("pieChart");
    var myPieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(typeCounts),
            datasets: [{
                data: Object.values(typeCounts),
                backgroundColor: ['#007bff', '#dc3545', '#ffc107', '#6838D1', '#F85A3E', '#84CAE7', '#000000'],
            }],
        },
    });
}

function addPredictionDescreiptions(typeCounts) {
    var maxKey = "";
    var maxValue = 0;
    $.each(typeCounts, function (key, val) {
        if (val > maxValue && key != "Normal") {
            maxKey = key;
            maxValue = val;
        }
    })

    switch (maxKey) {
        case "Premature Ventricular Contraction":
            $('#PVCinfo').show();
            break;
        case "Atrial Premature Contraction":
            $('#APCinfo').show();
            break;
        case "Left Bundle Branch Block":
            $('#LBBBinfo').show();
            break;
        case "Right Bundle Branch Block":
            $('#RBBBinfo').show();
            break;
        case "Ventricular Escape":
            $('#VEinfo').show();
            break;
        default: 
            $('#Ninfo').show();
            break;
    }
}
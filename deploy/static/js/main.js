$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('.results').hide();
    $('#btn-predict-image').hide();
    $('#btn-predict-audio').hide();
    $('#btn-retrieval').hide();


    
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('.imagePreview').hide();
                
                $('.audioPreview').hide();
                $('#btn-predict-image').show();
                $('#btn-predict-audio').show();
                $('#btn-retrieval').show();
                $('#imagePreview').fadeIn(650);
                $('#audioPreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
            
        }
    }
    
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict-image').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $("#audioUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict-audio').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $("#imageUploadretr").change(function () {
        $('.image-section').show();
        $('.row results').hide();
        $('#btn-retrieval').show();
        readURL(this);
    });

    // Predict audio
    $('#btn-predict-audio').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict_audio',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

    // Predict image
    $('#btn-predict-image').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict_image',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

    // Retrieval
    $('#btn-retrieval').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/retrieval',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                console.log("loading");
                $('.loader').hide();
                $('.results').fadeIn(600);
                $('#imagePreview1').html('<img src="/static/results/out_1.png">');
                $('#imagePreview2').html('<img src="/static/results/out_2.png">');
                $('#imagePreview3').html('<img src="/static/results/out_3.png">');
                $('#imagePreview4').html('<img src="/static/results/out_4.png">');
                $('#imagePreview5').html('<img src="/static/results/out_5.png">');
                $('#imagePreview6').html('<img src="/static/results/out_6.png">');
                $('#imagePreview7').html('<img src="/static/results/out_7.png">');
                $('#imagePreview8').html('<img src="/static/results/out_8.png">');
                $('#imagePreview9').html('<img src="/static/results/out_9.png">');
                console.log('Success!');
            },
        });
    });

});

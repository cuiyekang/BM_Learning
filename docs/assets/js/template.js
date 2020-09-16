jQuery(document).ready(function($) {
	$(".navbar").load("in_navbar.html");
	$("#footer").load("in_footer.html");
	$(".headroom").headroom({
		"tolerance": 20,
		"offset": 50,
		"classes": {
			"initial": "animated",
			"pinned": "slideDown",
			"unpinned": "slideUp"
		}
	});

	
});
---
layout: post
title: About
image: /images/about/1.jpg
---

I am a Ph.D. student at Università della Svizzera Italiana (Lugano, Switzerland). 
I research graph neural networks and their applications to systems that change over time.
<br>
I am also the main developer of <a href="https://danielegrattarola.github.io/spektral">Spektral</a>, a library for graph deep learning in Keras. 
<br>
<br>
Currently I am working with the <a href="http://www.neurontobrainlaboratory.ca">Neuron To Brain Lab</a> on the detection and localization of epileptic seizures.
<br>
<br>
I love coding, writing, traveling, food, art, cinema, reading, and plants.

<br>
Favourite places: Eivissa | Kyōto | Queensland.

<br>
<br>

<center>
    <span class='personal-social-media'>
        <a target="_blank" href="https://twitter.com/riceasphait" rel="noopener">
            <i class="fab fa-twitter" style="font-size: 30px;"></i>
        </a>
        <a target="_blank" href="https://github.com/danielegrattarola" rel="noopener">
            <i class="fab fa-github" style="font-size: 30px;"></i>
        </a>
        </a>
        <a target="_blank" href="https://linkedin.com/in/danielegrattarola" rel="noopener">
            <i class="fab fa-linkedin" style="font-size: 30px;"></i>
        </a>
        <a target="_blank" href="/feed.xml">
            <i class="fas fa-rss" style="font-size: 30px;"></i>
        </a>
    </span>
</center>

<br>

<center class="image-grid">
    <img src="/images/about/1.jpg" style="grid-column: 1 / span 2;" title="Île Saint-Louis, Paris, France">
    <img src="/images/about/2.jpg" style="grid-column: 1; overflow:hidden;" title="Heart Reef, Whitsundays, Australia">
    <img src="/images/about/3.jpg" style="grid-column: 2;" title="Es Vedrà, Ibiza, Spain">
    <img src="/images/about/4.jpg" style="grid-column: 1 / span 2;" title="Fushimi Inari-taisha, Kyoto, Japan">
    <img src="/images/about/5.jpg" style="grid-column: 1 / span 2;" title="3D TORONTO sign, Toronto, Canada">
</center>

<!-- SVG-->
<script type="text/javascript">
/*
 * Replace all SVG images with inline SVG
 */
jQuery('img.svg').each(function(){
    var $img = jQuery(this);
    var imgID = $img.attr('id');
    var imgClass = $img.attr('class');
    var imgURL = $img.attr('src');

    jQuery.get(imgURL, function(data) {
        // Get the SVG tag, ignore the rest
        var $svg = jQuery(data).find('svg');

        // Add replaced image's ID to the new SVG
        if(typeof imgID !== 'undefined') {
            $svg = $svg.attr('id', imgID);
        }
        // Add replaced image's classes to the new SVG
        if(typeof imgClass !== 'undefined') {
            $svg = $svg.attr('class', imgClass+' replaced-svg');
        }

        // Remove any invalid XML tags as per http://validator.w3.org
        $svg = $svg.removeAttr('xmlns:a');

        // Replace image with new SVG
        $img.replaceWith($svg);

    }, 'xml');

});

</script>

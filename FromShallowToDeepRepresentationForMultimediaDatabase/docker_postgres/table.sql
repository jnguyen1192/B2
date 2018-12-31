-- Table: public.image

-- DROP TABLE public.image;

CREATE TABLE public.image
(
  name_image text NOT NULL,
  des text,
  CONSTRAINT pk_image_name_image PRIMARY KEY (name_image)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.image
  OWNER TO postgres;

-- SELECT*FROM countries

-- CREATE TABLE cities(
--     name text NOT NULL,
--     postal_code varchar(9),
--     country_code char(2) REFERENCES countries,
--     PRIMARY KEY (country_code, postal_code)
-- );

-- INSERT INTO cities
-- VALUES ('Portland','97200','US');

-- UPDATE cities
-- SET postal_code = '97206'
-- WHERE name = 'Portland'

-- SELECT cities.*, country_name
-- FROM cities INNER JOIN countries
-- ON cities.country_code = countries.country_code
-- /* combines values from the 2 tables*/

-- CREATE TABLE venues(
--     venue_id SERIAL PRIMARY KEY,
--     name varchar(255),
--     street_address text,
--     type char(7) CHECK (type in ('public','private')) DEFAULT 'public',
--     postal_code varchar(9),
--     country_code char(2),
--     FOREIGN KEY (country_code, postal_code)
--     REFERENCES cities (country_code, postal_code) MATCH FULL
-- );


-- INSERT INTO venues (name, postal_code, country_code)
-- VALUES ('crystal ballroom', '97206', 'US')


SELECT v.venue_id, v.name, c.name
FROM venues v INNER JOIN cities c
ON v.postal_code =c.postal_code AND v.country_code = c.country_code;